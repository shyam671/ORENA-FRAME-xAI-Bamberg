"""GSViT vision encoder, exposed as a plain spatial feature extractor.

Like ``internvideo3_vision_tower`` (and unlike ``siglip_vision_tower`` / ``lemonfm_vision_tower``)
this module is **not** a drop-in ``model.model.visual``: it has no merger and emits a
``[n, C, g, g]`` feature map for ``multi_vision_tower`` to fuse and project.

GSViT (https://github.com/SamuelSchmidgall/GSViT, "General Surgery Vision Transformer") is an
EfficientViT-M5 initialised from ImageNet and then pre-trained on ~680h of general-surgery video
with a next-frame-prediction objective. The release is a bare ``OrderedDict`` holding

  * ``evit.*``    -- the encoder: ``nn.Sequential(patch_embed, blocks1, blocks2, blocks3)``
  * ``decoder.*`` -- the next-frame decoder head, discarded here.

Three things about the upstream setup are load-bearing:

* **Preprocessing is BGR in [0, 1], with no mean/std.** ``dataloader_surgical.py`` feeds raw
  ``cv2`` frames divided by 255, and ``load_gsvit.py::process_inputs`` swaps channels 0 and 2 on
  the way in, so an RGB tensor must be flipped. Hence ``GSVIT_MEAN``/``GSVIT_STD`` are the
  identity and ``gsvit_feature_map`` does the flip.
* **Key names are the original Microsoft EfficientViT ones** (``0.0.c.weight``, ``2.3.mixer...``)
  while timm's ``efficientvit_msra`` uses ``patch_embed.conv1.conv.weight`` /
  ``stages.1.blocks.0.mixer...``. Both state dicts enumerate the network in the same forward
  order, so ``load_gsvit_backbone`` maps them positionally and asserts every shape agrees --
  the same strategy timm's own ``checkpoint_filter_fn`` uses for this family.
* **The window sizes are baked into the 224 pretraining resolution.** ``LocalWindowAttention``
  takes ``window_resolution = min(7, stage_resolution)``, so at 224 the three stages use 7, 7 and
  4 (the checkpoint's attention-bias tables are 49, 49 and 16 entries). The model is therefore
  *built* at 224 and then retargeted with ``set_gsvit_input_size``, which only rewrites the
  ``resolution`` attribute the forward pass asserts on -- the trained window partitioning is
  preserved and the bias tables still match.

Geometry: patch_embed is stride 16 and the two ``PatchMerging`` stages halve again, so the final
stage is stride 64 at 384 channels -- at the 512 reference input that is an 8x8 map, which
``MultiVisionTowerWithMerger._feature_map`` bilinear-upsamples to the fused grid.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

GSVIT_DEFAULT_WEIGHTS = "/data/local/hf/hub/pretrained-vision-encoder/GSViT.pkl"
GSVIT_TIMM_MODEL = "efficientvit_m5"
GSVIT_HIDDEN = 384           # EfficientViT-M5 stage-3 width
GSVIT_STRIDE = 64            # patch_embed 16 x two PatchMerging stages
GSVIT_PRETRAIN_SIZE = 224    # the resolution the window sizes / bias tables were trained at
# Raw [0, 1] pixels, no mean/std: see the module docstring.
GSVIT_MEAN = (0.0, 0.0, 0.0)
GSVIT_STD = (1.0, 1.0, 1.0)

_WEIGHT_PREFIX = "evit."
# Resolution-derived index buffers; timm rebuilds them at construction, so they are not transferred.
_SKIP_SUFFIX = "attention_bias_idxs"


def set_gsvit_input_size(backbone: nn.Module, input_size: int) -> None:
    """Retarget the window attention to `input_size` (see the module docstring).

    Two attributes have to move together. ``resolution`` is only what ``forward`` asserts the
    incoming map size is. ``window_resolution`` is the partition size -- and timm leaves it at the
    nominal 7 even where the constructor clamped the attention-bias table to ``min(7, resolution)``,
    which is exactly the last stage (4 at 224). Feeding a larger input then takes the windowed
    branch with 7x7 windows against a 4x4 bias table. The trained window is recovered from the bias
    table itself rather than recomputed from 224, so this stays correct for any release.
    """
    from timm.models.efficientvit_msra import LocalWindowAttention

    for stage_idx, stage in enumerate(backbone.stages):
        resolution = input_size // (16 * 2 ** stage_idx)
        if resolution < 1:
            raise ValueError(
                f"GSViT input size {input_size} is too small: stage {stage_idx} would have "
                f"resolution {resolution}. Use a multiple of {GSVIT_STRIDE}."
            )
        for module in stage.modules():
            if isinstance(module, LocalWindowAttention):
                module.resolution = resolution
                module.window_resolution = round(module.attn.attention_bias_idxs.shape[0] ** 0.5)


def _keep_eval_while_frozen(backbone: nn.Module) -> None:
    """Make ``backbone.train(True)`` a no-op for as long as none of its parameters are trainable.

    EfficientViT is built out of ``Conv2d_BN``, so a *frozen* GSViT left in train mode would still
    update its BatchNorm running statistics on every forward pass and drift away from the
    pretrained weights the ablation is trying to measure. ``freeze_vision_tower`` only clears
    ``requires_grad``, and the tower's ``train()`` recurses into every child, so the guard has to
    live here. It re-arms automatically: unfreeze the encoder and ``train(True)`` behaves normally.
    """
    module_train = backbone.train

    def train(mode: bool = True):
        module_train(mode)
        if mode and not any(p.requires_grad for p in backbone.parameters()):
            module_train(False)
        return backbone

    backbone.train = train


def load_gsvit_backbone(
    weights_path: str = GSVIT_DEFAULT_WEIGHTS,
    input_size: int = GSVIT_PRETRAIN_SIZE,
) -> nn.Module:
    """EfficientViT-M5 with the GSViT weights loaded, retargeted to `input_size`."""
    import timm

    # Built at the pretraining resolution so the window sizes and attention-bias tables match the
    # checkpoint; set_gsvit_input_size then points the forward pass at the real input.
    backbone = timm.create_model(GSVIT_TIMM_MODEL, pretrained=False, num_classes=0)
    _keep_eval_while_frozen(backbone)

    if not os.path.isfile(weights_path):
        # Reload path (e.g. eval of a finetuned checkpoint): the trained weights come from the
        # checkpoint shards via `_load_checkpoint_weights_into`, so the release is not required.
        print(f"[GSViT] weights not found at {weights_path!r}; leaving the backbone uninitialised.")
        set_gsvit_input_size(backbone, input_size)
        return backbone

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    released = {
        key[len(_WEIGHT_PREFIX):]: value
        for key, value in checkpoint.items()
        if key.startswith(_WEIGHT_PREFIX) and not key.endswith(_SKIP_SUFFIX)
    }

    target = backbone.state_dict()
    if len(released) != len(target):
        raise RuntimeError(
            f"GSViT checkpoint has {len(released)} encoder tensors but timm's "
            f"{GSVIT_TIMM_MODEL} expects {len(target)}; the positional mapping is not safe."
        )
    remapped = {}
    for (timm_key, timm_value), (released_key, released_value) in zip(target.items(), released.items()):
        if tuple(timm_value.shape) != tuple(released_value.shape):
            raise RuntimeError(
                f"GSViT shape mismatch: {timm_key} is {tuple(timm_value.shape)} but "
                f"{released_key} is {tuple(released_value.shape)}."
            )
        remapped[timm_key] = released_value
    backbone.load_state_dict(remapped, strict=True)

    set_gsvit_input_size(backbone, input_size)
    return backbone


def gsvit_feature_map(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """``[n, 3, S, S]`` RGB in [0, 1] -> ``[n, 384, S/64, S/64]``."""
    # GSViT was pre-trained on cv2 frames, i.e. BGR (see the module docstring).
    return backbone.forward_features(pixel_values.flip(1))
