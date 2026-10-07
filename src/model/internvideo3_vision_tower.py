"""InternVideo3 vision encoder, exposed as a plain spatial feature extractor.

Unlike ``siglip_vision_tower`` / ``lemonfm_vision_tower`` this module is **not** a drop-in
``model.model.visual``: it has no merger and emits a ``[n, C, g, g]`` feature map for
``multi_vision_tower`` to concatenate with the other encoders. Only the vision half of
InternVideo3 (https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo3) is used; the
released vision shard contains exactly ``model.visual.{patch_embed,pos_embed,blocks.*}`` and no
merger weights, which is precisely the pre-merger backbone we want.

Two things about the upstream code are worth knowing:

* Its ``modeling_internvideo3.py`` is written against transformers 4.57 and does
  ``from transformers.utils.generic import check_model_inputs``, which no longer exists in the
  transformers this repo pins. That symbol only decorates the *text* / full models, never
  ``InternVideo3VisionModel``, so ``_install_check_model_inputs_shim`` installs a no-op before the
  dynamic import rather than us vendoring a copy of the 65k-line file.
* The encoder is Qwen2/3-VL-shaped: it consumes *flattened* patches ordered by 2x2 merge window
  (``t, gh/m, gw/m, m, m``) plus a ``grid_thw``, not ``[n, 3, S, S]``. ``patchify`` reproduces that
  layout and ``to_spatial_map`` inverts it. ``temporal_patch_size`` is 2, so each frame is
  duplicated into both temporal slots of the Conv3d stem -- consistent with how every other path in
  this repo treats video (independent frames).
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn

INTERNVIDEO3_DEFAULT_DIR = "/data/local/hf/hub/internvideo3-vision"
INTERNVIDEO3_HIDDEN = 1152          # vision_config.hidden_size
INTERNVIDEO3_PATCH = 16             # vision_config.patch_size
INTERNVIDEO3_TEMPORAL_PATCH = 2     # vision_config.temporal_patch_size
INTERNVIDEO3_MEAN = (0.5, 0.5, 0.5)
INTERNVIDEO3_STD = (0.5, 0.5, 0.5)

_WEIGHT_PREFIX = "model.visual."


def _install_check_model_inputs_shim() -> None:
    """No-op stand-in for the ``check_model_inputs`` decorator dropped in newer transformers.

    Idempotent; must run before the InternVideo3 remote module is imported.
    """
    import transformers.utils.generic as generic

    if hasattr(generic, "check_model_inputs"):
        return

    def check_model_inputs(*args, **kwargs):
        def decorator(fn):
            return fn

        # Support both `@check_model_inputs` and `@check_model_inputs()`.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return decorator

    generic.check_model_inputs = check_model_inputs


def load_internvideo3_backbone(model_dir: str = INTERNVIDEO3_DEFAULT_DIR) -> nn.Module:
    """``InternVideo3VisionModel`` with its own merger stripped and the released weights loaded."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _install_check_model_inputs_shim()

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vision_config = config.vision_config
    # The blocks pick their attention kernel off the config; sdpa needs no extra install and
    # handles the cu_seqlens-derived mask for our uniform single-frame grids.
    vision_config._attn_implementation = "sdpa"

    vision_cls = get_class_from_dynamic_module(
        "modeling_internvideo3.InternVideo3VisionModel", model_dir
    )
    backbone = vision_cls(vision_config)
    # We fuse before projecting, so InternVideo3's own 1152*4 -> 4096 merger is dead weight (and
    # unused parameters upset DeepSpeed). Identity keeps its `forward` returning per-patch features.
    backbone.merger = nn.Identity()
    backbone.deepstack_merger_list = nn.ModuleList()

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        # Reload path (e.g. eval of a finetuned checkpoint): the trained weights come from the
        # checkpoint shards via `_load_checkpoint_weights_into`, so the release is not required.
        print(f"[InternVideo3] no *.safetensors in {model_dir!r}; leaving the backbone uninitialised.")
        return backbone

    from safetensors.torch import load_file

    state_dict = {}
    for shard in shards:
        for key, value in load_file(shard).items():
            if key.startswith(_WEIGHT_PREFIX):
                state_dict[key[len(_WEIGHT_PREFIX):]] = value

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[InternVideo3] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    return backbone


def patchify(pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n, 3, S, S]`` -> ``(flat_patches, grid_thw)`` in InternVideo3's expected layout.

    Mirrors ``Qwen2VLImageProcessor``: patches are emitted grouped by 2x2 merge window, i.e. in
    ``(t, gh/m, gw/m, m, m)`` order, each carrying ``temporal_patch_size`` copies of the frame.
    """
    n, _, height, width = pixel_values.shape
    p, tp, m = INTERNVIDEO3_PATCH, INTERNVIDEO3_TEMPORAL_PATCH, 2
    if height != width or height % (p * m) != 0:
        raise ValueError(
            f"InternVideo3 needs square inputs divisible by {p * m} "
            f"(patch {p} x the 2x2 spatial merge); got {height}x{width}."
        )
    g = height // p

    x = pixel_values.unsqueeze(1).repeat(1, tp, 1, 1, 1)  # [n, tp, 3, S, S]
    x = x.view(n, tp, 3, g // m, m, p, g // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = x.reshape(n * g * g, 3 * tp * p * p)
    grid_thw = torch.tensor([[1, g, g]] * n, device=pixel_values.device)
    return flat, grid_thw


def to_spatial_map(hidden: torch.Tensor, n: int, g: int) -> torch.Tensor:
    """``[n*g*g, C]`` in merge-window order -> ``[n, C, g, g]`` in raster order."""
    m = 2
    c = hidden.shape[-1]
    x = hidden.view(n, g // m, g // m, m, m, c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(n, g, g, c)
    return x.permute(0, 3, 1, 2).contiguous()
