"""SurgeNetXL vision encoder, exposed as a plain spatial feature extractor.

Like ``internvideo3_vision_tower`` / ``gsvit_vision_tower`` this is **not** a drop-in
``model.model.visual``: it has no merger and emits a ``[n, C, g, g]`` feature map for
``multi_vision_tower`` to fuse and project.

SurgeNetXL (https://github.com/TimJaspers0801/SurgeNet) is a CAFormer-S18 trained with DINO on a
large multi-procedure surgical video corpus. The release is a DINO checkpoint holding ``student``
/ ``teacher`` / optimizer state; we take the **teacher**'s ``backbone.*`` keys (the EMA weights,
which is what the paper evaluates) and drop the projection head.

Two conveniences come from the checkpoint itself rather than from guesswork:

* ``checkpoint["args"].arch == "caformer_s18"`` pins the architecture, and its ``data_mean`` /
  ``data_std`` are the surgical-corpus statistics used during pretraining -- reproduced below as
  ``SURGENET_MEAN`` / ``SURGENET_STD`` rather than falling back to ImageNet's.
* The keys are in the original sail-sg ``metaformer_baselines`` layout
  (``downsample_layers.* / stages.<i>.<j>.* / norm.*``), which is exactly what
  ``timm.models.metaformer.checkpoint_filter_fn`` was written to remap onto timm's
  ``stem / stages.<i>.blocks.<j> / head.norm``.

The only keys timm then reports missing are the ``StarReLU`` ``scale`` / ``bias`` pairs: the
sail-sg CAFormer configuration keeps them non-trainable so they were never saved, and timm's
defaults (1.0 / 0.0) are the same constants. Anything else missing is a real mismatch and raises.

``head.norm`` is the trained output LayerNorm. Upstream it runs after global pooling; here it is a
``LayerNorm2d``, so applying it to the ``[n, C, h, w]`` map normalises **per position** -- the same
treatment ``lemonfm_vision_tower`` gives ConvNeXt's ``classifier[0]``.

Geometry: the stem is stride 4 and the three downsample layers halve again, so the final stage is
stride 32 at 512 channels -- at the 512 reference input that lands natively on a 16x16 map.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

SURGENET_DEFAULT_WEIGHTS = (
    "/data/local/hf/hub/pretrained-vision-encoder/SurgeNetXL_checkpoint0050.pth?download=true.1"
)
SURGENET_TIMM_MODEL = "caformer_s18"
SURGENET_HIDDEN = 512        # CAFormer-S18 stage-4 width
SURGENET_STRIDE = 32         # stem 4 x three downsample layers
# From the checkpoint's own args.data_mean / args.data_std (the SurgeNet corpus statistics).
SURGENET_MEAN = (0.46888983, 0.29536288, 0.28712815)
SURGENET_STD = (0.24689102, 0.21034359, 0.21188641)

_WEIGHT_PREFIX = "backbone."
# StarReLU's scale/bias are frozen constants upstream, so the checkpoint has no entry for them.
_ALLOWED_MISSING_SUFFIX = (".scale", ".bias")
_ALLOWED_MISSING_INFIX = ("act",)


def load_surgenet_backbone(weights_path: str = SURGENET_DEFAULT_WEIGHTS) -> nn.Module:
    """CAFormer-S18 with the SurgeNetXL DINO teacher weights loaded (projection head removed)."""
    import timm

    backbone = timm.create_model(
        SURGENET_TIMM_MODEL, pretrained=False, num_classes=0, drop_path_rate=0.0
    )

    if not os.path.isfile(weights_path):
        # Reload path (e.g. eval of a finetuned checkpoint): the trained weights come from the
        # checkpoint shards via `_load_checkpoint_weights_into`, so the release is not required.
        print(f"[SurgeNet] weights not found at {weights_path!r}; leaving the backbone uninitialised.")
        return backbone

    from timm.models.metaformer import checkpoint_filter_fn

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    teacher = checkpoint.get("teacher", checkpoint)
    state_dict = {
        key[len(_WEIGHT_PREFIX):]: value
        for key, value in teacher.items()
        if key.startswith(_WEIGHT_PREFIX)
    }
    if not state_dict:
        raise RuntimeError(f"No {_WEIGHT_PREFIX!r} keys in the SurgeNet checkpoint {weights_path!r}.")

    missing, unexpected = backbone.load_state_dict(
        checkpoint_filter_fn(state_dict, backbone), strict=False
    )
    unexplained = [
        key
        for key in missing
        if not (key.endswith(_ALLOWED_MISSING_SUFFIX) and any(i in key for i in _ALLOWED_MISSING_INFIX))
    ]
    if unexplained or unexpected:
        raise RuntimeError(
            f"SurgeNet checkpoint does not match timm's {SURGENET_TIMM_MODEL}: "
            f"missing={unexplained[:8]} unexpected={list(unexpected)[:8]}"
        )
    return backbone


def surgenet_feature_map(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """``[n, 3, S, S]`` -> ``[n, 512, S/32, S/32]``, output LayerNorm applied per position."""
    return backbone.head.norm(backbone.forward_features(pixel_values))
