"""Dispatch between the supported swapped vision encoders.

``--vision_encoder_id`` selects the encoder that replaces the native Qwen vision tower:
  * a SigLIP/SigLIP2 hub id (e.g. ``google/siglip2-so400m-patch16-512``)  -> ``siglip_vision_tower``
  * ``lemonfm[:<weights.pth>][@<image_size>]``                            -> ``lemonfm_vision_tower``
  * ``multi[:<enc>[+<enc>...]][@<grid>[,<flag>...]]``                     -> ``multi_vision_tower``
    (several encoders fused channel-wise before one merger; ``<enc>`` is ``siglip2[=<hf_id>]``,
    ``lemonfm[=<weights.pth>]`` or ``internvideo3[=<model_dir>]``; ``<flag>`` is ``prenorm``,
    which LayerNorms each stream before the concat, ``pixelshuffle``, which space-to-depths an
    oversized feature map onto the grid instead of area-averaging it, or ``merger=<spec>``, which
    picks the projector's shape -- see ``model/merger_variants.py``)

All three model loaders and the two preprocessing paths (training dataset, generation preprocessor)
go through the helpers here so a new encoder only has to be registered in one place.
"""

from __future__ import annotations

from model.lemonfm_vision_tower import is_lemonfm
from model.multi_vision_tower import is_multi


def build_vision_tower(qwen_config, encoder_id: str | None = None):
    """Build the swapped tower for `encoder_id`, sized to a loaded Qwen-VL config."""
    if is_multi(encoder_id):
        from model.multi_vision_tower import build_multi_vision_tower

        return build_multi_vision_tower(qwen_config, encoder_id)

    if is_lemonfm(encoder_id):
        from model.lemonfm_vision_tower import build_lemonfm_vision_tower

        return build_lemonfm_vision_tower(qwen_config, encoder_id)

    from model.siglip_vision_tower import build_siglip_vision_tower

    return build_siglip_vision_tower(qwen_config, encoder_id)


def load_vision_encoder_assets(encoder_id: str):
    """Return ``(image_processor, vision_config)`` for the swapped encoder.

    The image processor is called as ``ip(images=..., return_tensors="pt")["pixel_values"]`` and the
    vision config only needs ``.image_size`` / ``.patch_size`` (the fixed-resolution grid the
    dataset uses to size its ``<|image_pad|>`` expansion).
    """
    if is_multi(encoder_id):
        from model.multi_vision_tower import (
            MultiImageProcessor,
            multi_vision_config,
            parse_multi_id,
        )

        encoders, grid, flags = parse_multi_id(encoder_id)
        return MultiImageProcessor(), multi_vision_config(encoders, grid, flags)

    if is_lemonfm(encoder_id):
        from model.lemonfm_vision_tower import (
            LemonFMImageProcessor,
            lemonfm_vision_config,
            parse_lemonfm_id,
        )

        _, image_size = parse_lemonfm_id(encoder_id)
        return LemonFMImageProcessor(image_size), lemonfm_vision_config(image_size)

    from transformers import AutoConfig, AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    config = AutoConfig.from_pretrained(encoder_id)
    return image_processor, getattr(config, "vision_config", None) or config
