"""SigLIP2 vision tower that drops into a Qwen3-VL model as ``model.model.visual``.

The finetuning stack addresses the vision encoder through a small, implicit contract:
``model.model.visual`` is called (inside ``Qwen3VLModel.get_image_features``) as
``visual(pixel_values, grid_thw=..., return_dict=True)`` and must return an object with
``.pooler_output`` (the flattened per-merged-patch embeddings at the LLM hidden size) plus
``.deepstack_features``; it must also expose ``.dtype`` and ``.spatial_merge_size``, and hold a
``.merger`` submodule so freeze flags / per-group LRs work (``configure_vision_tower`` and the
optimizer key off the substrings ``"visual"`` / ``"merger"`` in parameter names).

Qwen3-VL's native tower emits variable-length ``grid_thw`` tokens with a deepstack multi-scale
path. SigLIP has neither, so we:
  * run standard SigLIP on fixed ``[n, 3, S, S]`` images (S/patch = grid_hw patches per side),
  * apply a ``spatial_merge_size`` x ``spatial_merge_size`` pixel-unshuffle + MLP ``merger`` and
    project to the LLM hidden size, matching a synthesized ``grid_thw = (t, grid_hw, grid_hw)``
    so ``grid.prod // spatial_merge_size**2 == emitted tokens`` and the M-RoPE / placeholder-mask
    math stay consistent, and
  * report an empty deepstack (``deepstack_features = []`` / ``deepstack_visual_indexes = []``),
    which the Qwen3-VL text layers treat as a no-op.

The module is encoder-agnostic: the vision width, image size and patch size are read from the
loaded SigLIP config, so any SigLIP/SigLIP2 checkpoint that produces a clean (CLS-free) square
patch grid works -- e.g. SigLIP2-Large-300M (hidden 1024) or SigLIP2-SO-400M (hidden 1152).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

# Configurable via --vision_encoder_id; this is only the fallback. The "-naflex" variants are
# NOT supported here (they need attention masks + spatial shapes); use a fixed-resolution one.
DEFAULT_SIGLIP_ENCODER = "google/siglip2-so400m-patch16-512"


class SiglipMerger(nn.Module):
    """spatial_merge x spatial_merge pixel-unshuffle then an MLP to the LLM hidden size."""

    def __init__(self, vision_hidden_size: int, out_hidden_size: int, spatial_merge_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.in_dim = vision_hidden_size * (spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(self.in_dim)
        self.fc1 = nn.Linear(self.in_dim, out_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_hidden_size, out_hidden_size)

    def forward(self, hidden_states: torch.Tensor, grid_hw: int) -> torch.Tensor:
        # hidden_states: [n, grid_hw*grid_hw, vision_hidden]
        n, _, c = hidden_states.shape
        m = self.spatial_merge_size
        merged_hw = grid_hw // m
        # [n, H, W, C] -> group m x m spatial neighbours into the channel dim (row-major).
        x = hidden_states.view(n, grid_hw, grid_hw, c)
        x = x.view(n, merged_hw, m, merged_hw, m, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(n, merged_hw * merged_hw, m * m * c)
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x  # [n, merged_hw*merged_hw, out_hidden_size]


class SiglipVisionTowerWithMerger(nn.Module):
    """Drop-in replacement for ``Qwen3VLModel.visual`` backed by a SigLIP2 encoder."""

    # Marker so shared helpers (e.g. the no-image dummy path) can branch on the tower type.
    is_siglip = True

    def __init__(self, encoder_id: str, out_hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        self.encoder_id = encoder_id
        self.spatial_merge_size = spatial_merge_size
        self.vision_model = SiglipVisionModel.from_pretrained(encoder_id).vision_model
        vcfg = self.vision_model.config
        self.vision_hidden = vcfg.hidden_size
        self.image_size = vcfg.image_size
        self.patch_size = vcfg.patch_size
        self.grid_hw = self.image_size // self.patch_size  # patches per side (e.g. 512 // 16 = 32)
        self.merger = SiglipMerger(self.vision_hidden, out_hidden_size, spatial_merge_size)
        # Qwen3-VL deepstack contract: empty -> the text layers skip all injection.
        self.deepstack_visual_indexes: list[int] = []

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def blocks(self):
        # Alias so unfreeze_topk_layers (backbone.visual.blocks[-k:]) keeps working.
        return self.vision_model.encoder.layers

    def enable_vision_gradient_checkpointing(self) -> None:
        """Shared hook name across swapped towers (see train_sft.configure/gradient_checkpointing)."""
        self.vision_model.gradient_checkpointing_enable()

    def make_dummy_inputs(self):
        """Zero image + grid for the no-image DeepSpeed path (keeps the vision graph connected)."""
        dummy_pixel = torch.zeros(
            (1, 3, self.image_size, self.image_size), device=self.device, dtype=self.dtype
        )
        dummy_grid = torch.tensor([[1, self.grid_hw, self.grid_hw]], device=self.device)
        return dummy_pixel, dummy_grid

    def forward(self, pixel_values: torch.Tensor, grid_thw=None, return_dict: bool = True, **kwargs):
        # pixel_values: [n, 3, S, S] (one row per image, or per video frame).
        pixel_values = pixel_values.to(self.dtype)
        vision_out = self.vision_model(pixel_values=pixel_values)
        hidden = vision_out.last_hidden_state  # [n, grid_hw**2, vision_hidden]
        merged = self.merger(hidden, self.grid_hw)  # [n, (grid_hw/m)**2, out_hidden_size]
        pooled = merged.reshape(-1, merged.shape[-1])  # [n*tokens_per_image, out_hidden_size]
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden,
            pooler_output=pooled,
            deepstack_features=[],
        )


def build_siglip_vision_tower(qwen_config, encoder_id: str | None = None):
    """Construct a SigLIP tower sized to a loaded Qwen3-VL config."""
    encoder_id = encoder_id or DEFAULT_SIGLIP_ENCODER
    vision_config = getattr(qwen_config, "vision_config", None)
    out_hidden_size = getattr(vision_config, "out_hidden_size", None)
    if out_hidden_size is None:
        text_config = getattr(qwen_config, "text_config", qwen_config)
        out_hidden_size = text_config.hidden_size
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2) if vision_config else 2
    return SiglipVisionTowerWithMerger(
        encoder_id=encoder_id,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
    )
