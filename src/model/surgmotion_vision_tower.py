"""SurgMotion vision encoder, exposed as a plain spatial feature extractor.

Like ``internvideo3_vision_tower`` / ``surgenet_vision_tower`` this is **not** a drop-in
``model.model.visual``: it has no merger and emits a ``[n, C, g, g]`` feature map for
``multi_vision_tower`` to fuse and project.

SurgMotion (https://arxiv.org/abs/2602.05638) is a **V-JEPA 2** video encoder pretrained on
SurgMotion-15M (3,658 h of surgical video). The release ships two variants and a `config.json`
that names only the backbone, so the architecture was recovered from the checkpoint itself:

* every tensor sits under ``module.backbone.*`` (a DDP-wrapped save) in the original timm-ish
  layout: ``patch_embed.proj`` is a ``Conv3d(3, C, (2,16,16), stride (2,16,16))`` -- tubelet 2,
  patch 16 -- then ``blocks.<i>.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}``, then ``norm``;
* there is **no** ``cls_token`` and **no** ``pos_embed``, because V-JEPA 2 positions tokens with
  parameter-free 3D RoPE;
* ``attn.qkv.bias`` has a non-zero k-third, which rules out VideoMAE (it stores split
  ``q_bias`` / ``v_bias`` and a zero k-bias). This is a plain fused-qkv ViT.

Rather than vendoring a ViT we load into ``transformers``' ``VJEPA2Encoder``, which already
implements the 3D RoPE and the tubelet stem. Only the key names differ, and only in ways a literal
rename plus a qkv split fixes, so the remap below loads ``strict=True``.

Two properties of that class make this integration cheap:

* ``VJEPA2RopeAttention`` derives its position grid from ``crop_size // patch_size``, and RoPE
  extrapolates across grid sizes. Setting ``crop_size`` to the tower's per-encoder input size means
  the encoder lands on the fused grid natively -- no feature-map resampling at all.
* ``VJEPA2Embeddings.forward`` duplicates a single frame up to ``tubelet_size`` on its own, which is
  exactly the convention ``internvideo3_vision_tower`` already documents: each frame fills both
  temporal slots of the Conv3d stem, consistent with how every other path in this repo treats video
  (independent frames). SurgMotion therefore contributes its *spatial* representation here; the
  frame-based data pipeline never hands it real motion.

**num_attention_heads is not recoverable from the weights** -- qkv shapes are identical for any head
count, so a wrong value would load cleanly via ``strict=True`` and silently compute the wrong
attention. The presets below are V-JEPA 2's published configs (``facebook/vjepa2-vitl-*`` and
``facebook/vjepa2-vitg-*``); an unrecognised backbone raises rather than guesses.

Normalisation is V-JEPA 2's own (``VJEPA2VideoProcessor`` defaults to ImageNet mean/std). Unlike
SurgeNetXL's DINO checkpoint, the SurgMotion release carries no corpus-specific statistics.
"""

from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn as nn

from model.lemonfm_vision_tower import IMAGENET_MEAN, IMAGENET_STD

SURGMOTION_ROOT = "/data/local/hf/hub/pretrained-vision-encoder/SurgMotion"
SURGMOTION_VITL_DIR = f"{SURGMOTION_ROOT}/SurgMotion-vitl"
SURGMOTION_VITG_DIR = f"{SURGMOTION_ROOT}/SurgMotion-vitg"
SURGMOTION_PATCH = 16          # spatial patch -> the input size is PATCH * grid
SURGMOTION_TUBELET = 2         # temporal patch; one frame fills both slots
SURGMOTION_VITL_HIDDEN = 1024  # V-JEPA 2 ViT-L width
SURGMOTION_VITG_HIDDEN = 1408  # V-JEPA 2 ViT-g width
SURGMOTION_MEAN = IMAGENET_MEAN
SURGMOTION_STD = IMAGENET_STD

# config.json's `backbone` -> V-JEPA 2's published architecture. See the docstring on head counts.
_VARIANTS = {
    "vit_large": dict(
        hidden_size=SURGMOTION_VITL_HIDDEN, num_hidden_layers=24, num_attention_heads=16, mlp_ratio=4.0
    ),
    "vit_giant_xformer": dict(
        hidden_size=SURGMOTION_VITG_HIDDEN, num_hidden_layers=40, num_attention_heads=22, mlp_ratio=48 / 11
    ),
}

_WEIGHT_PREFIX = "module.backbone."


def _remap_to_hf(state_dict: dict[str, torch.Tensor], hidden_size: int) -> dict[str, torch.Tensor]:
    """SurgMotion's ``module.backbone.*`` names -> ``VJEPA2Encoder``'s, splitting the fused qkv."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(_WEIGHT_PREFIX):
            continue
        key = key[len(_WEIGHT_PREFIX):]
        if key.startswith("patch_embed.proj."):
            remapped[f"embeddings.patch_embeddings.proj.{key.rsplit('.', 1)[1]}"] = value
        elif key.startswith("norm."):
            remapped[f"layernorm.{key.rsplit('.', 1)[1]}"] = value
        elif key.startswith("blocks."):
            _, index, rest = key.split(".", 2)
            if rest.startswith("attn.qkv."):
                # HF keeps q/k/v as three Linears; the release fuses them into one [3C, C] / [3C].
                suffix = rest.rsplit(".", 1)[1]
                for name, chunk in zip(("query", "key", "value"), value.split(hidden_size, dim=0)):
                    remapped[f"layer.{index}.attention.{name}.{suffix}"] = chunk
            elif rest.startswith("attn.proj."):
                remapped[f"layer.{index}.attention.proj.{rest.rsplit('.', 1)[1]}"] = value
            else:  # norm1 / norm2 / mlp.fc1 / mlp.fc2 are named identically
                remapped[f"layer.{index}.{rest}"] = value
    return remapped


def load_surgmotion_backbone(model_dir: str = SURGMOTION_VITL_DIR, input_size: int = 256) -> nn.Module:
    """``VJEPA2Encoder`` sized to the SurgMotion variant in `model_dir`, with its weights loaded.

    `input_size` sets ``crop_size``, which is what the RoPE grid is derived from, so it must match
    the resolution the tower actually feeds this encoder.
    """
    from transformers import VJEPA2Config
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"[SurgMotion] no config.json in {model_dir!r}; cannot size the encoder.")
    backbone_name = json.load(open(config_path))["backbone"]
    if backbone_name not in _VARIANTS:
        raise ValueError(
            f"[SurgMotion] unknown backbone {backbone_name!r} in {config_path!r}; "
            f"expected one of {sorted(_VARIANTS)}. Add its V-JEPA 2 preset rather than guessing "
            f"num_attention_heads -- the weight shapes do not pin it down."
        )
    preset = _VARIANTS[backbone_name]

    config = VJEPA2Config(
        crop_size=input_size,
        frames_per_clip=SURGMOTION_TUBELET,
        patch_size=SURGMOTION_PATCH,
        tubelet_size=SURGMOTION_TUBELET,
        qkv_bias=True,
        **preset,
    )
    # sdpa needs no extra install and handles our uniform single-tubelet grids.
    config._attn_implementation = "sdpa"
    # The encoder only: V-JEPA 2's 12-layer predictor is dead weight here (we fuse before
    # projecting) and unused parameters upset DeepSpeed.
    backbone = VJEPA2Encoder(config)

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        # Reload path (e.g. eval of a finetuned checkpoint): the trained weights come from the
        # checkpoint shards via `_load_checkpoint_weights_into`, so the release is not required.
        print(f"[SurgMotion] no *.safetensors in {model_dir!r}; leaving the backbone uninitialised.")
        return backbone

    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(shard))
    remapped = _remap_to_hf(state_dict, preset["hidden_size"])
    if not remapped:
        raise RuntimeError(f"No {_WEIGHT_PREFIX!r} keys in the SurgMotion release under {model_dir!r}.")

    # strict: the remap is exact, so a mismatch means the release changed layout -- fail loudly.
    backbone.load_state_dict(remapped, strict=True)
    print(f"[SurgMotion] loaded {backbone_name} from {model_dir!r} ({len(remapped)} tensors).")
    return backbone


def surgmotion_feature_map(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """``[n, 3, S, S]`` -> ``[n, C, S/16, S/16]``, output LayerNorm applied per position."""
    n, _, size, _ = pixel_values.shape
    g = size // SURGMOTION_PATCH
    # [n, T=1, 3, S, S]: `VJEPA2Embeddings` repeats the lone frame across the tubelet itself.
    hidden = backbone(pixel_values_videos=pixel_values.unsqueeze(1)).last_hidden_state
    return hidden.transpose(1, 2).reshape(n, -1, g, g)
