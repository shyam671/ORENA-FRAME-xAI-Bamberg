from __future__ import annotations

import glob
import os
from typing import Any

from transformers import AutoConfig, AutoModelForImageTextToText, PretrainedConfig

from model.modeling_cls import (
    Qwen2VLForSequenceClassification,
    Qwen2_5_VLForSequenceClassification,
    Qwen3VLForSequenceClassification,
    Qwen3_5ForSequenceClassification,
    Qwen3_5MoeForSequenceClassification,
)
from train.monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen3_5_moe_with_mixed_modality_forward,
    replace_qwen3_5_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
)
from train.monkey_patch_vision import replace_qwen2_5_vision

_GENERATION_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
}

_PATCHERS = {
    "qwen2_vl": (replace_qwen_2_with_mixed_modality_forward,),
    "qwen2_5_vl": (
        replace_qwen2_5_with_mixed_modality_forward,
        replace_qwen2_5_vision,
    ),
    "qwen3_5": (replace_qwen3_5_with_mixed_modality_forward,),
    "qwen3_5_moe": (replace_qwen3_5_moe_with_mixed_modality_forward,),
    "qwen3_vl": (replace_qwen3_with_mixed_modality_forward,),
    "qwen3_vl_moe": (replace_qwen3_vl_moe_with_mixed_modality_forward,),
}

_SEQUENCE_CLASSIFICATION_MODEL_CLS = {
    "qwen2_vl": Qwen2VLForSequenceClassification,
    "qwen2_5_vl": Qwen2_5_VLForSequenceClassification,
    "qwen3_5": Qwen3_5ForSequenceClassification,
    "qwen3_5_moe": Qwen3_5MoeForSequenceClassification,
    "qwen3_vl": Qwen3VLForSequenceClassification,
}


def get_qwen_vl_generation_backbone(model):
    if not hasattr(model, "model"):
        raise TypeError(f"Unsupported generation model wrapper: {type(model)!r}")
    return model.model


def apply_qwen_vl_monkey_patches(model_type: str) -> str:
    try:
        patchers = _PATCHERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_PATCHERS))
        raise ValueError(f"Unsupported Qwen-VL model_type: {model_type}. Supported: {supported}") from exc

    for patcher in patchers:
        patcher()

    return model_type


def _load_checkpoint_weights_into(model, model_path: str) -> None:
    """Load any local checkpoint weights into `model` with strict=False.

    Used after swapping in a SigLIP tower on reload: the monolithic `from_pretrained`
    cannot match the SigLIP `visual.*` keys against the native Qwen tower it built, so
    those weights are dropped. Once the SigLIP tower is in place we re-load the shards
    to populate the finetuned `visual.*` (LLM keys reload harmlessly).
    """
    if not os.path.isdir(model_path):
        return
    state_dict = {}
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        from safetensors.torch import load_file

        for f in safetensor_files:
            state_dict.update(load_file(f))
    else:
        import torch

        for f in sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin"))):
            state_dict.update(torch.load(f, map_location="cpu"))
    if state_dict:
        model.load_state_dict(state_dict, strict=False)


def maybe_swap_vision_encoder(model, config, vision_encoder_id: str | None, model_path: str | None):
    """Replace the native Qwen vision tower with a swapped tower (SigLIP2 / LemonFM) when requested.

    `vision_encoder_id` (an explicit override) takes precedence; otherwise a
    `config.vision_encoder_id` persisted at save time triggers the same swap on reload.
    The id selects the encoder -- see `model.vision_encoder.build_vision_tower`.
    On reload of a finetuned checkpoint we also re-load the local shards so the trained
    encoder/merger weights land in the freshly-built tower.
    """
    encoder_id = vision_encoder_id or getattr(config, "vision_encoder_id", None)
    if not encoder_id:
        return model

    from model.vision_encoder import build_vision_tower

    backbone = get_qwen_vl_generation_backbone(model)
    tower = build_vision_tower(config, encoder_id)
    # Match the tower to the rest of the model (bf16/CPU at train start, fp16/CUDA at inference).
    ref = model.get_input_embeddings().weight
    tower = tower.to(device=ref.device, dtype=ref.dtype)
    backbone.visual = tower
    model.config.vision_encoder_id = encoder_id

    # If the checkpoint already carried this flag, its shards hold the trained tower weights.
    if getattr(config, "vision_encoder_id", None) and model_path is not None:
        _load_checkpoint_weights_into(model, model_path)

    return model


def load_qwen_vl_generation_model(
    model_name_or_path: str,
    *,
    config: PretrainedConfig | None = None,
    vision_encoder_id: str | None = None,
    **kwargs: Any,
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type not in _GENERATION_MODEL_TYPES:
        supported = ", ".join(sorted(_GENERATION_MODEL_TYPES))
        raise ValueError(
            f"Unsupported Qwen-VL generation model_type: {config.model_type}. Supported: {supported}"
        )

    apply_qwen_vl_monkey_patches(config.model_type)
    # Reloading a swapped checkpoint: its config carries vision_encoder_id and its shards hold
    # SigLIP `visual.*` weights that will not match the native tower from_pretrained builds.
    # Tolerate the mismatch here; maybe_swap_vision_encoder then installs the SigLIP tower and
    # re-loads the trained weights into it.
    if getattr(config, "vision_encoder_id", None):
        kwargs.setdefault("ignore_mismatched_sizes", True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )
    model = maybe_swap_vision_encoder(model, config, vision_encoder_id, model_name_or_path)
    return model


def get_qwen_vl_sequence_classification_model_cls(model_type: str):
    try:
        return _SEQUENCE_CLASSIFICATION_MODEL_CLS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_SEQUENCE_CLASSIFICATION_MODEL_CLS))
        raise ValueError(
            f"Unsupported Qwen-VL sequence classification model_type: {model_type}. Supported: {supported}"
        ) from exc


def load_qwen_vl_sequence_classification_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    apply_qwen_vl_monkey_patches(config.model_type)
    model_cls = get_qwen_vl_sequence_classification_model_cls(config.model_type)
    return model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )
