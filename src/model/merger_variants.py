"""Merger (projector) architecture variants for the fused multi-encoder tower.

The merger is the only randomly-initialised module in a ``multi:`` tower and the only thing that
mixes the sub-encoders, so its shape bounds what the fusion can express. Every run up to v27 used
one fixed shape -- ``SiglipMerger``: a 2x2 pixel-unshuffle, one LayerNorm over the fused vector,
``fc1(in_dim -> out_hidden)``, GELU, ``fc2(out_hidden -> out_hidden)``. This module parameterises
that shape so it can be swept.

Selected by a ``merger=<spec>`` flag on the encoder id, e.g.::

    multi:siglip2=...+lemonfm=...@16,merger=ps1-swiglu-fc1

``<spec>`` is ``<reduce>-<act>-<depth>``:

    <reduce> := ps2  2x2 pixel-unshuffle (space-to-depth); fc1 sees 4x the channels   -> (g/2)^2 tokens
              | ps1  no spatial reduction; fc1 sees the raw fused width               -> g^2   tokens
              | ap2  2x2 average-pool; ps2's token count at ps1's fc1 width           -> (g/2)^2 tokens
    <act>    := gelu | silu | swiglu
    <depth>  := fc2   fc1(in -> out), act, fc2(out -> out)      -- the current shape
              | fc1   fc1(in -> out), act                       -- no second layer
              | wide  fc1(in -> in),  act, fc2(in -> out)       -- Qwen3VLVisionPatchMerger's geometry

``swiglu`` means ``fc1`` emits ``2 * hidden``, split into ``(gate, up)`` with output
``SiLU(gate) * up``; that doubles ``fc1``'s parameters relative to the plain-``silu`` sibling, which
is inherent to the gate and is not compensated for.

``DEFAULT_VARIANT`` ("ps2-gelu-fc2") is the shape every prior run used. ``build_variant_merger``
returns ``None`` for it so the caller keeps constructing the untouched ``SiglipMerger`` /
``FusedStreamMerger``, which keeps old checkpoints and the no-flag id bit-identical.

**Token-count contract.** ``ps1`` emits 4x the tokens per frame. The dataset and the generation
preprocessor both size their ``<|image_pad|>`` expansion as
``(image_size // patch_size)**2 // spatial_merge_size**2`` and Qwen3-VL splits the emitted features
with ``grid_thw.prod() // visual.spatial_merge_size**2``, so ``visual.spatial_merge_size`` must stay
at Qwen's 2 and the *reported* ``patch_size`` absorbs the merger's reduction instead -- which is
what ``merger_patch_size`` below computes for ``multi_vision_tower.multi_vision_config``.

REVERTING
---------
Everything variant-specific is in this file. To drop the sweep entirely::

    rm src/model/merger_variants.py scripts/check_merger_variants.py \\
       scripts/finetune_merger_sweep_v28.sh
    grep -n '\\[v28\\]' src/model/multi_vision_tower.py     # 5 marked hunks, each self-describing

No other file references this module: no CLI argument, no config key, no change to the dataset, the
trainer or the eval harness. Checkpoints written without a ``merger=`` flag do not depend on it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.siglip_vision_tower import SiglipMerger

# The shape every run up to v27 used; `build_variant_merger` declines to build it so the caller
# falls back to the original class and nothing about the default path changes.
DEFAULT_VARIANT = "ps2-gelu-fc2"

MERGER_FLAG_PREFIX = "merger="

# Qwen3-VL's `config.vision_config.spatial_merge_size`. The dataset divides by it to size the
# <|image_pad|> expansion and Qwen splits the emitted features by it, so the tower keeps reporting
# this value and a merger with a different reduction is absorbed by `patch_size` instead.
QWEN_SPATIAL_MERGE = 2

# reduce -> spatial factor applied before the projection
_REDUCERS = {"ps2": 2, "ps1": 1, "ap2": 2}
_ACTS = {"gelu": nn.GELU, "silu": nn.SiLU, "swiglu": nn.SiLU}
_DEPTHS = ("fc2", "fc1", "wide")


def parse_merger_variant(variant: str) -> tuple[str, str, str]:
    """``"ps1-swiglu-fc1"`` -> ``("ps1", "swiglu", "fc1")``, validated."""
    parts = variant.strip().lower().split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Merger variant must be <reduce>-<act>-<depth>, got {variant!r}; "
            f"e.g. {DEFAULT_VARIANT!r}."
        )
    reduce_, act, depth = parts
    for field, value, allowed in (
        ("reduce", reduce_, sorted(_REDUCERS)),
        ("activation", act, sorted(_ACTS)),
        ("depth", depth, sorted(_DEPTHS)),
    ):
        if value not in allowed:
            raise ValueError(
                f"Unknown merger {field} {value!r} in {variant!r}; expected one of {allowed}."
            )
    return reduce_, act, depth


def merger_variant_from_flags(flags: set[str]) -> str:
    """The ``merger=<spec>`` entry of a parsed encoder-id flag set, or the default."""
    picked = sorted(f for f in flags if f.startswith(MERGER_FLAG_PREFIX))
    if len(picked) > 1:
        raise ValueError(f"At most one merger= flag is allowed; got {picked}.")
    if not picked:
        return DEFAULT_VARIANT
    variant = picked[0][len(MERGER_FLAG_PREFIX):]
    parse_merger_variant(variant)
    return variant


def merger_reduction(variant: str) -> int:
    """Spatial factor the merger applies, i.e. ``grid / tokens_per_side``."""
    return _REDUCERS[parse_merger_variant(variant)[0]]


def merger_patch_size(reference_size: int, grid: int, flags: set[str]) -> int:
    """The ``patch_size`` a vision-config stand-in must report for this merger.

    The consumers compute ``(image_size // patch_size)**2 // QWEN_SPATIAL_MERGE**2`` pads per frame
    and the merger emits ``(grid / merger_reduction)**2`` tokens, so the two agree only if the
    reported patch size carries the merger's reduction. For the default merger this is
    ``reference_size // grid``, i.e. exactly what the caller used before this module existed.
    """
    tokens_side = grid // merger_reduction(merger_variant_from_flags(flags))
    if reference_size % (tokens_side * QWEN_SPATIAL_MERGE) != 0:
        raise ValueError(
            f"Grid {grid} and this merger leave {tokens_side} tokens per side, which needs a "
            f"non-integer patch size in the config stand-in ({reference_size} / "
            f"{tokens_side * QWEN_SPATIAL_MERGE})."
        )
    return reference_size // (tokens_side * QWEN_SPATIAL_MERGE)


class VariantMerger(SiglipMerger):
    """``SiglipMerger`` with the reduction, activation and depth taken from a variant spec.

    Subclasses ``SiglipMerger`` so the parameters keep the names ``merger.norm/fc1/fc2``: both
    ``train_sft.configure_vision_tower`` (``--freeze_merger``) and ``QwenSFTTrainer.create_optimizer``
    (``--merger_lr``) key off the literal substring ``"merger"`` in the parameter path.
    """

    def __init__(self, vision_hidden_size: int, out_hidden_size: int, variant: str):
        reduce_, act, depth = parse_merger_variant(variant)
        # Only `ps2` folds the 2x2 neighbourhood into the channel dim; `ap2` averages it away and
        # `ps1` leaves the grid alone, so both project from the raw fused width.
        super().__init__(vision_hidden_size, out_hidden_size, 2 if reduce_ == "ps2" else 1)
        self.variant = variant
        self.reduction = _REDUCERS[reduce_]
        self.avg_pool = reduce_ == "ap2"
        self.gated = act == "swiglu"
        self.two_layer = depth != "fc1"

        hidden = self.in_dim if depth == "wide" else out_hidden_size
        self.act = _ACTS[act]()
        self.fc1 = nn.Linear(self.in_dim, 2 * hidden if self.gated else hidden)
        if self.two_layer:
            self.fc2 = nn.Linear(hidden, out_hidden_size)
        else:
            del self.fc2

    def forward(self, hidden_states: torch.Tensor, grid_hw: int) -> torch.Tensor:
        # hidden_states: [n, grid_hw*grid_hw, vision_hidden]
        n, _, c = hidden_states.shape
        r = self.reduction
        side = grid_hw // r

        if self.avg_pool:
            x = hidden_states.view(n, grid_hw, grid_hw, c).permute(0, 3, 1, 2)
            x = F.avg_pool2d(x, r).flatten(2).transpose(1, 2)      # [n, side**2, c]
        elif r > 1:
            # Same row-major grouping as SiglipMerger: m x m spatial neighbours -> channels.
            x = hidden_states.view(n, side, r, side, r, c)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(n, side * side, r * r * c)
        else:
            x = hidden_states

        x = self.norm(x)
        h = self.fc1(x)
        if self.gated:
            gate, up = h.chunk(2, dim=-1)
            h = self.act(gate) * up
        else:
            h = self.act(h)
        return self.fc2(h) if self.two_layer else h   # [n, side**2, out_hidden_size]


class VariantStreamMerger(VariantMerger):
    """``VariantMerger`` plus ``prenorm``'s per-encoder LayerNorm before the channel concat.

    Mirrors ``multi_vision_tower.FusedStreamMerger`` so a variant still composes with the
    ``prenorm`` flag. Defined here rather than inherited from it to keep the import one-way.
    """

    def __init__(self, stream_dims: list[int], out_hidden_size: int, variant: str):
        super().__init__(sum(stream_dims), out_hidden_size, variant)
        self.stream_norms = nn.ModuleList([nn.LayerNorm(c) for c in stream_dims])

    def fuse(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """``[n, g*g, C_i]`` per encoder -> normalised channel concat ``[n, g*g, sum(C_i)]``."""
        return torch.cat([norm(x) for norm, x in zip(self.stream_norms, streams)], dim=-1)


def build_variant_merger(
    variant: str,
    stream_dims: list[int],
    out_hidden_size: int,
    prenorm: bool = False,
    spatial_merge_size: int = QWEN_SPATIAL_MERGE,
) -> nn.Module | None:
    """The merger for ``variant``, or ``None`` for the default so the caller keeps the old class."""
    if variant == DEFAULT_VARIANT:
        return None
    if spatial_merge_size != QWEN_SPATIAL_MERGE:
        raise ValueError(
            f"merger={variant} assumes Qwen's spatial_merge_size {QWEN_SPATIAL_MERGE}; this config "
            f"reports {spatial_merge_size}, which would desynchronise the emitted token count from "
            "the dataset's <|image_pad|> expansion."
        )
    cls = VariantStreamMerger if prenorm else VariantMerger
    arg = stream_dims if prenorm else sum(stream_dims)
    return cls(arg, out_hidden_size, variant)
