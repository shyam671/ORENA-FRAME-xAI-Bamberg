"""LemonFM vision tower that drops into a Qwen3-VL model as ``model.model.visual``.

LemonFM (https://huggingface.co/visurg/LemonFM) is a surgical foundation model: a torchvision
ConvNeXt-Large trained with DINO on surgical video. Its checkpoint is a raw ``.pth`` holding
``{"teacher": ..., "student": ...}``; the ``teacher``'s ``backbone.*`` keys load into
``torchvision.models.convnext_large`` with ``classifier[2]`` replaced by ``nn.Identity()``
(1536-d features, total stride 32, ImageNet normalisation, pretrained at 224x224).

It satisfies the same implicit contract as ``siglip_vision_tower`` (see that module's docstring):
``model.model.visual`` is called as ``visual(pixel_values, grid_thw=..., return_dict=True)`` and
returns ``.pooler_output`` (flattened per-merged-patch embeddings at the LLM hidden size) plus an
empty ``.deepstack_features``; it exposes ``.dtype`` / ``.spatial_merge_size`` / ``.blocks`` and
holds a submodule literally named ``merger`` so the freeze flags and per-group LRs keep working.

Geometry: ConvNeXt is a plain conv net, so ``features(x)`` on ``[n, 3, S, S]`` yields
``[n, 1536, S/32, S/32]``. Qwen3-VL's M-RoPE uses ``config.vision_config.spatial_merge_size`` (2)
while the token split uses ``visual.spatial_merge_size``, so the two must agree: the merge stays
2x2 and the feature map must be even-sided, i.e. **S must be a multiple of 64**. LemonFM's native
224 (7x7, odd) is therefore not usable directly; 448 -> 14x14 -> 49 tokens/image and
640 -> 20x20 -> 100 tokens/image are.

The encoder id is self-describing so training and eval can never drift:
``lemonfm:/data/local/hf/hub/lemonfm.pth@448`` (both parts optional; see ``parse_lemonfm_id``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from model.siglip_vision_tower import SiglipMerger

LEMONFM_DEFAULT_WEIGHTS = "/data/local/hf/hub/pretrained-vision-encoder/lemonfm.pth"
LEMONFM_DEFAULT_IMAGE_SIZE = 448
LEMONFM_STRIDE = 32          # ConvNeXt-Large: 4x stem + 3 downsamples
LEMONFM_HIDDEN = 1536        # final-stage width
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_LEMONFM_ALIASES = {"lemonfm", "visurg/lemonfm"}


def is_lemonfm(encoder_id: str | None) -> bool:
    """True for ids like ``lemonfm``, ``lemonfm:/path/to.pth@448`` or ``visurg/LemonFM``."""
    if not encoder_id:
        return False
    scheme = encoder_id.split(":", 1)[0].split("@", 1)[0].strip().lower()
    return scheme in _LEMONFM_ALIASES


def parse_lemonfm_id(encoder_id: str) -> tuple[str, int]:
    """``lemonfm[:<weights_path>][@<image_size>]`` -> ``(weights_path, image_size)``."""
    spec = encoder_id.split(":", 1)[1] if ":" in encoder_id else ""
    if "@" in spec:
        spec, size_str = spec.rsplit("@", 1)
        image_size = int(size_str)
    elif "@" in encoder_id and not spec:
        image_size = int(encoder_id.rsplit("@", 1)[1])
    else:
        image_size = LEMONFM_DEFAULT_IMAGE_SIZE
    weights_path = spec.strip() or LEMONFM_DEFAULT_WEIGHTS

    if image_size % (LEMONFM_STRIDE * 2) != 0:
        raise ValueError(
            f"LemonFM image size must be a multiple of {LEMONFM_STRIDE * 2} "
            f"(ConvNeXt stride {LEMONFM_STRIDE} x the 2x2 spatial merge Qwen3-VL's M-RoPE assumes); "
            f"got {image_size} from {encoder_id!r}."
        )
    return weights_path, image_size


def load_lemonfm_backbone(weights_path: str) -> nn.Module:
    """ConvNeXt-Large with the LemonFM DINO teacher weights loaded (classifier head removed)."""
    from torchvision.models import convnext_large

    # weights=None, never "DEFAULT": no download, and the LemonFM weights overwrite everything.
    net = convnext_large(weights=None)
    net.classifier[2] = nn.Identity()

    if not os.path.isfile(weights_path):
        # Fail loudly. The previous behaviour was to warn and return the randomly initialised net
        # (on the theory that a reloaded finetuned checkpoint gets its tower weights from the
        # shards via `_load_checkpoint_weights_into` anyway) -- but a typo'd path landed here too
        # and trained a random ConvNeXt-Large without anyone noticing, which silently invalidated
        # the whole v9/v10 sweep. Never guess; the caller must point at the real .pth.
        raise FileNotFoundError(
            f"[LemonFM] no weights at {weights_path!r}. Refusing to return a randomly initialised "
            f"ConvNeXt-Large. Pass the real checkpoint via the encoder id, e.g. "
            f"lemonfm={LEMONFM_DEFAULT_WEIGHTS}"
        )

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("teacher", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    prefix = "backbone."
    state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    net.load_state_dict(state_dict, strict=True)
    print(f"[LemonFM] loaded DINO teacher weights from {weights_path!r} ({len(state_dict)} tensors).")
    return net


class LemonFMImageProcessor:
    """Duck-typed stand-in for ``AutoImageProcessor`` (fixed square, ImageNet normalisation).

    Kept call-compatible with the SigLIP path so ``get_siglip_image_pixels`` /
    ``get_siglip_video_pixels`` and the eval preprocessor need no changes: accepts a PIL image, a
    list of PIL images / arrays, or the uint8 ``[T, 3, H, W]`` tensor ``qwen_vl_utils.fetch_video``
    returns, and always yields ``{"pixel_values": [n, 3, S, S]}`` float32.
    """

    def __init__(self, image_size: int = LEMONFM_DEFAULT_IMAGE_SIZE):
        self.image_size = image_size
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    @staticmethod
    def _to_chw_tensor(image) -> torch.Tensor:
        """One image -> float ``[3, H, W]`` in [0, 255]."""
        if isinstance(image, torch.Tensor):
            t = image
        else:
            import numpy as np
            from PIL import Image

            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            t = torch.from_numpy(np.asarray(image).copy())
        if t.ndim == 2:                       # grayscale HW
            t = t.unsqueeze(-1).expand(-1, -1, 3)
        if t.ndim != 3:
            raise ValueError(f"Expected a single image, got shape {tuple(t.shape)}")
        if t.shape[0] not in (1, 3):          # HWC -> CHW
            t = t.permute(2, 0, 1)
        if t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        return t[:3].float()

    def _split(self, images):
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            return list(images)
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    def __call__(self, images=None, return_tensors="pt", **kwargs):
        from torchvision.transforms.functional import resize

        s = self.image_size
        batch = [self._to_chw_tensor(img) for img in self._split(images)]
        batch = [resize(t, [s, s], antialias=True) for t in batch]
        pixel_values = torch.stack(batch, dim=0) / 255.0
        pixel_values = (pixel_values - self.mean) / self.std
        return {"pixel_values": pixel_values}


class LemonFMVisionTowerWithMerger(nn.Module):
    """Drop-in replacement for ``Qwen3VLModel.visual`` backed by the LemonFM ConvNeXt encoder."""

    # Marker so shared helpers can branch on the tower type (mirrors SigLIP's `is_siglip`).
    is_lemonfm = True

    def __init__(
        self,
        encoder_id: str,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        weights_path, image_size = parse_lemonfm_id(encoder_id)
        self.encoder_id = encoder_id
        self.weights_path = weights_path
        self.spatial_merge_size = spatial_merge_size
        self.backbone = load_lemonfm_backbone(weights_path)
        self.vision_hidden = LEMONFM_HIDDEN
        self.image_size = image_size
        self.patch_size = LEMONFM_STRIDE
        self.grid_hw = image_size // LEMONFM_STRIDE  # e.g. 448 // 32 = 14
        self.merger = SiglipMerger(self.vision_hidden, out_hidden_size, spatial_merge_size)
        # Qwen3-VL deepstack contract: empty -> the text layers skip all injection.
        self.deepstack_visual_indexes: list[int] = []
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def blocks(self):
        # Alias so unfreeze_topk_layers (backbone.visual.blocks[-k:]) keeps working: every ConvNeXt
        # block across all four stages, in forward order.
        from torchvision.models.convnext import CNBlock

        return nn.ModuleList([m for m in self.backbone.features.modules() if isinstance(m, CNBlock)])

    def enable_vision_gradient_checkpointing(self) -> None:
        """Stage-wise activation checkpointing (torchvision ConvNeXt has no built-in toggle)."""
        self.gradient_checkpointing = True

    def make_dummy_inputs(self):
        """Zero image + grid for the no-image DeepSpeed path (keeps the vision graph connected)."""
        dummy_pixel = torch.zeros(
            (1, 3, self.image_size, self.image_size), device=self.device, dtype=self.dtype
        )
        dummy_grid = torch.tensor([[1, self.grid_hw, self.grid_hw]], device=self.device)
        return dummy_pixel, dummy_grid

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            hidden = pixel_values
            for stage in self.backbone.features:
                hidden = checkpoint(stage, hidden, use_reentrant=False)
            return hidden
        return self.backbone.features(pixel_values)

    def forward(self, pixel_values: torch.Tensor, grid_thw=None, return_dict: bool = True, **kwargs):
        # pixel_values: [n, 3, S, S] (one row per image, or per video frame).
        pixel_values = pixel_values.to(self.dtype)
        feat = self._encode(pixel_values)                # [n, 1536, grid_hw, grid_hw]
        feat = self.backbone.classifier[0](feat)         # trained LayerNorm2d, applied per position
        n, c, h, w = feat.shape
        hidden = feat.flatten(2).transpose(1, 2)         # [n, grid_hw**2, 1536]
        merged = self.merger(hidden, h)                  # [n, (grid_hw/m)**2, out_hidden_size]
        pooled = merged.reshape(-1, merged.shape[-1])    # [n*tokens_per_image, out_hidden_size]
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden,
            pooler_output=pooled,
            deepstack_features=[],
        )


def lemonfm_vision_config(image_size: int) -> SimpleNamespace:
    """Minimal vision-config stand-in: the dataset/eval only need image_size and patch_size."""
    return SimpleNamespace(
        image_size=image_size,
        patch_size=LEMONFM_STRIDE,
        hidden_size=LEMONFM_HIDDEN,
    )


def build_lemonfm_vision_tower(qwen_config, encoder_id: str | None = None):
    """Construct a LemonFM tower sized to a loaded Qwen3-VL config."""
    encoder_id = encoder_id or "lemonfm"
    vision_config = getattr(qwen_config, "vision_config", None)
    out_hidden_size = getattr(vision_config, "out_hidden_size", None)
    if out_hidden_size is None:
        text_config = getattr(qwen_config, "text_config", qwen_config)
        out_hidden_size = text_config.hidden_size
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2) if vision_config else 2
    return LemonFMVisionTowerWithMerger(
        encoder_id=encoder_id,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
    )
