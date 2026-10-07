"""Multi-encoder vision tower: several backbones fused channel-wise before a single merger.

Drops into a Qwen3-VL model as ``model.model.visual`` under the same implicit contract as
``siglip_vision_tower`` (see that module's docstring). The difference is that instead of one
backbone there are up to three, each run at its own preferred input resolution, each feature map
resampled to a common ``grid x grid`` patch grid, and the maps concatenated along the **channel**
dimension before the shared ``SiglipMerger`` projects to the LLM hidden size::

    input [n, 3, 512, 512], raw [0, 1]        (n = images, or video frames)
      SigLIP2-SO-400M  512^2 -> 32x32x1152 --bilinear--> grid x grid x 1152
      LemonFM          512^2 -> 16x16x1536               grid x grid x 1536
      InternVideo3     256^2 -> 16x16x1152               grid x grid x 1152
                                    cat(dim=1) -> grid x grid x 3840
                        SiglipMerger(3840, out_hidden_size, m=2)
                                    -> (grid/2)^2 tokens per frame

At the default ``grid=16`` that is 64 tokens/frame (cf. 256 for SigLIP2-512 alone, 49 for
LemonFM@448). Only the merger mixes the encoders, so it is randomly initialised and always trains.

**Normalisation lives in the tower, not the processor.** The training dataset and the eval
preprocessor both consume a single ``image_processor(images=...)["pixel_values"]`` tensor and a
vision config exposing only ``.image_size`` / ``.patch_size``. So ``MultiImageProcessor`` emits
*unnormalised* ``[0, 1]`` pixels at one reference size and ``forward`` does the per-encoder resize
and mean/std, which keeps the whole data pipeline unchanged. The config stand-in reports
``image_size = REFERENCE_SIZE`` and ``patch_size = REFERENCE_SIZE // grid`` so the dataset's
existing ``image_size // patch_size`` arithmetic lands on ``grid``.

A one-encoder set is legal and is how the single-encoder ablation sweep
(``scripts/arxiv/finetune_focus_encoder_sweep_v11.sh``) runs each backbone: going through this
tower rather than its own gives every arm the same reference input, the same fused grid and
therefore the same token budget, so the encoders are directly comparable.

Encoder id grammar (self-describing, persisted into the saved config so eval re-swaps identically)::

    multi[:<enc>[+<enc>...]][@<grid>[,<flag>...]]
    <enc>  := siglip2[=<hf_id>] | lemonfm[=<weights.pth>] | internvideo3[=<model_dir>]
            | gsvit[=<weights.pkl>] | surgenetxl[=<weights.pth>]
            | surgmotion_vitl[=<model_dir>] | surgmotion_vitg[=<model_dir>]
    <flag> := prenorm | pixelshuffle | merger=<reduce>-<act>-<depth>

``prenorm`` puts a per-encoder ``LayerNorm`` on each stream's tokens *before* the channel concat
(see ``FusedStreamMerger``); without it the streams are concatenated raw, exactly as before.

``pixelshuffle`` changes how an encoder whose feature map is *larger* than the fused grid is brought
down to it: a ``r x r`` pixel-unshuffle (space-to-depth, ``r = side // grid``) instead of the default
``adaptive_avg_pool2d``. Nothing is averaged away -- the r^2 positions move into the channel dim, so
that stream widens by r^2 (SigLIP2 at ``@16``: 32x32x1152 -> 16x16x4608) and the merger is sized
accordingly. Encoders that already land on the grid, or below it, are unaffected.

``merger=<spec>`` [v28] swaps the merger's own architecture -- its spatial reduction, activation and
depth. Omitted, the merger is the one every run up to v27 used. See ``model/merger_variants.py``
for the grammar and for how to revert the whole thing; note that a spec which drops the 2x2
reduction emits ``grid^2`` tokens per frame instead of ``(grid/2)^2``, which ``multi_vision_config``
absorbs into the ``patch_size`` it reports.

Examples: ``multi``, ``multi@16``, ``multi:siglip2+lemonfm@16``, ``multi:gsvit@16``,
``multi:surgmotion_vitg@16``,
``multi:siglip2+lemonfm@16,prenorm``, ``multi:siglip2+lemonfm@16,pixelshuffle``,
``multi:siglip2+internvideo3=/data/local/hf/hub/pretrained-vision-encoder/internvideo3-vision@16``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from model.gsvit_vision_tower import (
    GSVIT_DEFAULT_WEIGHTS,
    GSVIT_HIDDEN,
    GSVIT_MEAN,
    GSVIT_STD,
    GSVIT_STRIDE,
    gsvit_feature_map,
    load_gsvit_backbone,
)
from model.internvideo3_vision_tower import (
    INTERNVIDEO3_DEFAULT_DIR,
    INTERNVIDEO3_HIDDEN,
    INTERNVIDEO3_MEAN,
    INTERNVIDEO3_PATCH,
    INTERNVIDEO3_STD,
    load_internvideo3_backbone,
    patchify,
    to_spatial_map,
)
from model.lemonfm_vision_tower import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LEMONFM_DEFAULT_WEIGHTS,
    LEMONFM_HIDDEN,
    LEMONFM_STRIDE,
    LemonFMImageProcessor,
    load_lemonfm_backbone,
)
from model.merger_variants import (  # [v28] merger sweep; see that module's REVERTING note
    MERGER_FLAG_PREFIX,
    build_variant_merger,
    merger_patch_size,
    merger_variant_from_flags,
)
from model.siglip_vision_tower import DEFAULT_SIGLIP_ENCODER, SiglipMerger
from model.surgenet_vision_tower import (
    SURGENET_DEFAULT_WEIGHTS,
    SURGENET_HIDDEN,
    SURGENET_MEAN,
    SURGENET_STD,
    SURGENET_STRIDE,
    load_surgenet_backbone,
    surgenet_feature_map,
)
from model.surgmotion_vision_tower import (
    SURGMOTION_MEAN,
    SURGMOTION_PATCH,
    SURGMOTION_STD,
    SURGMOTION_VITG_DIR,
    SURGMOTION_VITG_HIDDEN,
    SURGMOTION_VITL_DIR,
    SURGMOTION_VITL_HIDDEN,
    load_surgmotion_backbone,
    surgmotion_feature_map,
)

# The two SurgMotion arms share every dispatch branch and differ only in the registry entry.
_SURGMOTION = ("surgmotion_vitl", "surgmotion_vitg")

# Every encoder is fed a resize of this square; it is also what the processor emits and what the
# dataset believes the "image size" is. 512 is SigLIP2-SO-400M's native resolution (its position
# embeddings are fixed), so it is the largest of the three and downsampling from it is lossless.
REFERENCE_SIZE = 512
DEFAULT_GRID = 16
DEFAULT_ENCODERS = ("siglip2", "lemonfm", "internvideo3")

_MULTI_ALIASES = {"multi", "multiencoder", "multi_encoder"}
_MULTI_FLAGS = {"prenorm", "pixelshuffle"}

# SigLIP2-SO-400M-patch16-512; the only encoder whose stride is not imported from its own module.
SIGLIP_PATCH = 16

# name -> (hidden dim, default source, input size given the fused grid)
# GSViT is the one encoder whose input size is not `stride * grid`: its deepest stage is stride 64,
# so landing natively on the grid would mean feeding it 1024 -- 4x the compute, and far from the 224
# it was pretrained at. It runs at the reference size instead and `_feature_map` upsamples its
# (grid/2)^2 map, which is the deepest-stage/least-compute end of that trade.
_ENCODER_SPECS = {
    "siglip2": (1152, DEFAULT_SIGLIP_ENCODER, lambda grid: REFERENCE_SIZE),
    "lemonfm": (LEMONFM_HIDDEN, LEMONFM_DEFAULT_WEIGHTS, lambda grid: LEMONFM_STRIDE * grid),
    "internvideo3": (INTERNVIDEO3_HIDDEN, INTERNVIDEO3_DEFAULT_DIR, lambda grid: INTERNVIDEO3_PATCH * grid),
    "gsvit": (GSVIT_HIDDEN, GSVIT_DEFAULT_WEIGHTS, lambda grid: (GSVIT_STRIDE // 2) * grid),
    "surgenetxl": (SURGENET_HIDDEN, SURGENET_DEFAULT_WEIGHTS, lambda grid: SURGENET_STRIDE * grid),
    # V-JEPA 2 positions tokens with RoPE, which extrapolates, so `crop_size` follows the grid and
    # SurgMotion lands on it natively (the loader is told the size for exactly that reason).
    "surgmotion_vitl": (SURGMOTION_VITL_HIDDEN, SURGMOTION_VITL_DIR, lambda grid: SURGMOTION_PATCH * grid),
    "surgmotion_vitg": (SURGMOTION_VITG_HIDDEN, SURGMOTION_VITG_DIR, lambda grid: SURGMOTION_PATCH * grid),
}
# name -> output stride of the feature map `_feature_map` reads, i.e. input size / feature side.
# Only `pixelshuffle` uses this: the merger has to be sized before any forward pass, so the shuffle
# factor must be derivable statically. `_feature_map` recomputes it from the real map and asserts the
# two agree, so a drift here fails loudly instead of mis-sizing the merger.
_ENCODER_STRIDE = {
    "siglip2": SIGLIP_PATCH,
    "lemonfm": LEMONFM_STRIDE,
    "internvideo3": INTERNVIDEO3_PATCH,
    "gsvit": GSVIT_STRIDE,
    "surgenetxl": SURGENET_STRIDE,
    "surgmotion_vitl": SURGMOTION_PATCH,
    "surgmotion_vitg": SURGMOTION_PATCH,
}
_ENCODER_NORM = {
    "siglip2": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "lemonfm": (IMAGENET_MEAN, IMAGENET_STD),
    "internvideo3": (INTERNVIDEO3_MEAN, INTERNVIDEO3_STD),
    "gsvit": (GSVIT_MEAN, GSVIT_STD),
    "surgenetxl": (SURGENET_MEAN, SURGENET_STD),
    "surgmotion_vitl": (SURGMOTION_MEAN, SURGMOTION_STD),
    "surgmotion_vitg": (SURGMOTION_MEAN, SURGMOTION_STD),
}


def is_multi(encoder_id: str | None) -> bool:
    """True for ids like ``multi``, ``multi@16`` or ``multi:siglip2+lemonfm@16``."""
    if not encoder_id:
        return False
    scheme = encoder_id.split(":", 1)[0].split("@", 1)[0].strip().lower()
    return scheme in _MULTI_ALIASES


def parse_multi_id(encoder_id: str) -> tuple[dict[str, str], int, set[str]]:
    """``multi[:<enc>[+<enc>...]][@<grid>[,<flag>...]]`` -> ``({encoder: source}, grid, flags)``.

    The encoder dict is insertion-ordered (it fixes the channel order of the concat, so it must be
    stable between training and eval -- which it is, since the id itself is persisted).
    """
    body = encoder_id.strip()
    flags: set[str] = set()
    if "@" in body:
        body, suffix = body.rsplit("@", 1)
        grid_str, *flag_strs = [part.strip().lower() for part in suffix.split(",")]
        grid = int(grid_str)
        flags = {flag for flag in flag_strs if flag}
        # [v28] `merger=<spec>` is the one key=value flag; validated eagerly so a typo fails here
        # rather than after the encoders have downloaded. Revert: drop the two merger clauses.
        unknown = {f for f in flags if f not in _MULTI_FLAGS and not f.startswith(MERGER_FLAG_PREFIX)}
        if unknown:
            raise ValueError(
                f"Unknown fusion flag(s) {sorted(unknown)} in {encoder_id!r}; "
                f"expected one of {sorted(_MULTI_FLAGS)} or {MERGER_FLAG_PREFIX}<spec>."
            )
        merger_variant_from_flags(flags)
    else:
        grid = DEFAULT_GRID

    spec = body.split(":", 1)[1].strip() if ":" in body else ""
    names = [part.strip() for part in spec.split("+") if part.strip()] if spec else list(DEFAULT_ENCODERS)

    if grid % 2 != 0:
        raise ValueError(
            f"Fused grid must be even (Qwen3-VL's M-RoPE assumes a 2x2 spatial merge); "
            f"got {grid} from {encoder_id!r}."
        )
    if REFERENCE_SIZE % grid != 0:
        raise ValueError(
            f"Fused grid must divide the reference size {REFERENCE_SIZE} so the vision config "
            f"stand-in can report an integer patch size; got {grid} from {encoder_id!r}."
        )

    encoders: dict[str, str] = {}
    for name in names:
        key, _, source = name.partition("=")
        key = key.strip().lower()
        if key not in _ENCODER_SPECS:
            raise ValueError(
                f"Unknown encoder {key!r} in {encoder_id!r}; expected one of {sorted(_ENCODER_SPECS)}."
            )
        if key in encoders:
            raise ValueError(f"Encoder {key!r} listed twice in {encoder_id!r}.")
        encoders[key] = source.strip() or _ENCODER_SPECS[key][1]
    if not encoders:
        raise ValueError(f"No encoders selected by {encoder_id!r}.")
    return encoders, grid, flags


def multi_shuffle_factors(encoders: dict[str, str], grid: int, flags: set[str]) -> list[int]:
    """Per-encoder pixel-unshuffle factor, in concat order; ``1`` means "no unshuffle".

    Without the ``pixelshuffle`` flag every factor is 1 and the tower behaves exactly as before.
    With it, an encoder whose native map is an integer multiple of the grid is space-to-depthed by
    that multiple (SigLIP2 at ``@16``: 32 // 16 = 2); one that lands on the grid, below it, or on a
    non-integer multiple keeps the default resample.
    """
    if "pixelshuffle" not in flags:
        return [1] * len(encoders)
    factors = []
    for name in encoders:
        side = _ENCODER_SPECS[name][2](grid) // _ENCODER_STRIDE[name]
        factors.append(side // grid if side > grid and side % grid == 0 else 1)
    return factors


def multi_stream_dims(encoders: dict[str, str], factors: list[int] | None = None) -> list[int]:
    """Per-encoder channel widths, in concat order; pixel-unshuffle widens a stream by ``r**2``."""
    dims = [_ENCODER_SPECS[name][0] for name in encoders]
    if factors is None:
        return dims
    return [c * r * r for c, r in zip(dims, factors)]


def multi_hidden_size(encoders: dict[str, str], factors: list[int] | None = None) -> int:
    """Channel width of the concatenated feature map."""
    return sum(multi_stream_dims(encoders, factors))


class MultiImageProcessor(LemonFMImageProcessor):
    """Emits raw ``[0, 1]`` pixels at the reference size; the tower normalises per encoder.

    Reuses LemonFM's duck-typed ``AutoImageProcessor`` shim (PIL / list / ``[T, 3, H, W]`` tensor
    handling) and only replaces the normalisation step.
    """

    def __init__(self, image_size: int = REFERENCE_SIZE):
        super().__init__(image_size)

    def __call__(self, images=None, return_tensors="pt", **kwargs):
        from torchvision.transforms.functional import resize

        s = self.image_size
        batch = [self._to_chw_tensor(img) for img in self._split(images)]
        batch = [resize(t, [s, s], antialias=True) for t in batch]
        return {"pixel_values": torch.stack(batch, dim=0) / 255.0}


def multi_vision_config(
    encoders: dict[str, str], grid: int, flags: set[str] | None = None
) -> SimpleNamespace:
    """Vision-config stand-in: ``image_size // patch_size == grid`` is all the dataset needs."""
    flags = flags or set()
    factors = multi_shuffle_factors(encoders, grid, flags)
    return SimpleNamespace(
        image_size=REFERENCE_SIZE,
        # [v28] was `REFERENCE_SIZE // grid`; identical unless a merger= flag changes the reduction.
        patch_size=merger_patch_size(REFERENCE_SIZE, grid, flags),
        hidden_size=multi_hidden_size(encoders, factors),
    )


class FusedStreamMerger(SiglipMerger):
    """``SiglipMerger`` with a per-encoder ``LayerNorm`` applied before the channel concat.

    The plain merger concatenates the streams raw and normalises the joint vector once, inside
    ``SiglipMerger.forward``. That single LayerNorm is computed over all ``sum(C_i)`` channels
    together, so it rescales the fused vector as a whole and does nothing to equalise the streams
    relative to each other: SigLIP2 (contrastive) and LemonFM (DINO-distilled ConvNeXt) features
    have very different magnitudes, and the larger one dominates the projection. Normalising each
    stream first removes that imbalance while leaving its per-channel structure intact.

    This lives inside the merger rather than on the tower on purpose: the freeze flags
    (``train_sft.configure_vision_tower``) and the per-group LRs (``QwenSFTTrainer.create_optimizer``)
    both key off the literal substring ``"merger"`` in the parameter name, so parameters named
    ``visual.merger.stream_norms.*`` train at ``--merger_lr`` and stay trainable in a
    merger-only stage where the encoders are frozen. Parameters hung off the tower directly would
    be frozen along with the encoders.
    """

    def __init__(self, stream_dims: list[int], out_hidden_size: int, spatial_merge_size: int):
        super().__init__(sum(stream_dims), out_hidden_size, spatial_merge_size)
        self.stream_norms = nn.ModuleList([nn.LayerNorm(c) for c in stream_dims])

    def fuse(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """``[n, g*g, C_i]`` per encoder -> normalised channel concat ``[n, g*g, sum(C_i)]``."""
        return torch.cat([norm(x) for norm, x in zip(self.stream_norms, streams)], dim=-1)


class MultiVisionTowerWithMerger(nn.Module):
    """Drop-in replacement for ``Qwen3VLModel.visual`` fusing several encoders channel-wise."""

    # Marker so shared helpers can branch on the tower type (mirrors `is_siglip` / `is_lemonfm`).
    is_multi = True

    def __init__(self, encoder_id: str, out_hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        encoders, grid, flags = parse_multi_id(encoder_id)
        self.encoder_id = encoder_id
        self.encoder_names = list(encoders)
        self.prenorm = "prenorm" in flags
        self.pixelshuffle = "pixelshuffle" in flags
        self.spatial_merge_size = spatial_merge_size
        self.image_size = REFERENCE_SIZE
        self.grid_hw = grid
        # [v28] was `REFERENCE_SIZE // grid`. Must stay equal to what `multi_vision_config` reports:
        # a merger= flag that changes the spatial reduction changes the grid the *consumers* see
        # (the <|image_pad|> expansion, Qwen's feature split), not the encoder's internal one.
        # Identical to the old expression unless such a flag is present.
        self.patch_size = merger_patch_size(REFERENCE_SIZE, grid, flags)
        # Under `pixelshuffle` an oversized map is space-to-depthed onto the grid instead of
        # area-averaged, which widens that stream by r^2; the merger is sized for it here.
        factors = multi_shuffle_factors(encoders, grid, flags)
        self.shuffle_factors = dict(zip(encoders, factors))
        self.stream_dims = multi_stream_dims(encoders, factors)
        self.vision_hidden = sum(self.stream_dims)
        self.gradient_checkpointing = False

        # Per-encoder input resolution, derived from the fused grid so each backbone lands on the
        # grid natively where it can (only SigLIP2, whose position embeddings are fixed at 512,
        # needs its feature map resampled).
        self.input_sizes = {name: _ENCODER_SPECS[name][2](grid) for name in encoders}

        for name, source in encoders.items():
            if name == "siglip2":
                from transformers import SiglipVisionModel

                self.siglip = SiglipVisionModel.from_pretrained(source).vision_model
                # We read `last_hidden_state`, so SigLIP's attention-pooling head never runs;
                # leaving it attached would hand DeepSpeed 11 parameters that get no gradient.
                if getattr(self.siglip, "use_head", False):
                    self.siglip.use_head = False
                    del self.siglip.head
            elif name == "lemonfm":
                self.lemonfm = load_lemonfm_backbone(source)
            elif name == "internvideo3":
                self.internvideo3 = load_internvideo3_backbone(source)
            elif name == "gsvit":
                self.gsvit = load_gsvit_backbone(source, self.input_sizes[name])
            elif name == "surgenetxl":
                self.surgenetxl = load_surgenet_backbone(source)
            elif name in _SURGMOTION:
                # Needs the input size like GSViT, but to set the RoPE grid rather than a window.
                setattr(self, name, load_surgmotion_backbone(source, self.input_sizes[name]))
            mean, std = _ENCODER_NORM[name]
            self.register_buffer(f"{name}_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
            self.register_buffer(f"{name}_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # [v28] a `merger=<spec>` flag swaps the projector's architecture; it builds nothing for the
        # default spec, so the original two branches below are reached exactly as before.
        # Revert: delete the two statements and the first branch here.
        self.merger_variant = merger_variant_from_flags(flags)
        variant_merger = build_variant_merger(
            self.merger_variant, self.stream_dims, out_hidden_size, self.prenorm, spatial_merge_size
        )
        if variant_merger is not None:
            self.merger = variant_merger
        elif self.prenorm:
            self.merger = FusedStreamMerger(
                self.stream_dims, out_hidden_size, spatial_merge_size
            )
        else:
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
        """Every encoder's blocks, interleaved **from the top**.

        ``unfreeze_topk_layers`` takes ``blocks[-k:]``; laying the encoders out end-to-end would
        make that unfreeze the tail of one encoder only, so instead position -1 is the last block
        of the first encoder, -2 the last block of the second, and so on. ``blocks[-k:]`` therefore
        unfreezes roughly the top ``k / n_encoders`` layers of *each* encoder.
        """
        from torchvision.models.convnext import CNBlock

        per_encoder = []
        for name in self.encoder_names:
            if name == "siglip2":
                per_encoder.append(list(self.siglip.encoder.layers))
            elif name == "lemonfm":
                per_encoder.append([m for m in self.lemonfm.features.modules() if isinstance(m, CNBlock)])
            elif name == "internvideo3":
                per_encoder.append(list(self.internvideo3.blocks))
            elif name == "gsvit":
                per_encoder.append([b for stage in self.gsvit.stages for b in stage.blocks])
            elif name == "surgenetxl":
                per_encoder.append([b for stage in self.surgenetxl.stages for b in stage.blocks])
            elif name in _SURGMOTION:
                per_encoder.append(list(getattr(self, name).layer))

        interleaved = []
        for depth in range(max(len(b) for b in per_encoder)):
            for blocks in per_encoder:
                if depth < len(blocks):
                    interleaved.append(blocks[-(depth + 1)])
        interleaved.reverse()
        return nn.ModuleList(interleaved)

    def encoder_module(self, name: str) -> nn.Module:
        """The submodule holding one encoder's weights (``siglip2`` is stored as ``self.siglip``)."""
        return getattr(self, "siglip" if name == "siglip2" else name)

    def set_trainable_encoders(self, keep: set[str]) -> None:
        """Freeze every sub-encoder not in ``keep``; the merger is untouched.

        Called *after* the whole-tower pass in ``train_sft.configure_vision_tower``, so this is an
        override of ``--freeze_vision_tower False`` rather than a second freezing policy. Note that
        ``unfreeze_topk_layers`` runs later still and spans every encoder (see ``blocks``), which is
        why ``train_sft`` rejects the two together.
        """
        unknown = keep - set(self.encoder_names)
        if unknown:
            raise ValueError(f"unknown encoder(s) {sorted(unknown)}; tower has {self.encoder_names}")
        for name in self.encoder_names:
            for p in self.encoder_module(name).parameters():
                p.requires_grad = name in keep

    def enable_vision_gradient_checkpointing(self) -> None:
        """Shared hook name across swapped towers (see train_sft.configure/gradient_checkpointing)."""
        self.gradient_checkpointing = True
        if "siglip2" in self.encoder_names:
            self.siglip.gradient_checkpointing_enable()
        if "internvideo3" in self.encoder_names:
            try:
                self.internvideo3.gradient_checkpointing_enable()
            except Exception as exc:  # remote code: tolerate an unsupported toggle
                print(f"[multi] InternVideo3 gradient checkpointing unavailable: {exc}")
        for name in ("gsvit", "surgenetxl"):
            if name in self.encoder_names:
                getattr(self, name).set_grad_checkpointing(True)  # timm's shared API
        # LemonFM's torchvision ConvNeXt has no built-in toggle; `_encode_lemonfm` checkpoints
        # stage-wise off `self.gradient_checkpointing`.
        # SurgMotion is absent on purpose: `VJEPA2Encoder.forward` has no checkpointing hook, and
        # while it is frozen none of its activations are kept for backward anyway. Unfreezing it
        # would mean checkpointing its `.layer` list by hand, as `_encode_lemonfm` does.

    @property
    def reported_grid_hw(self) -> int:
        """[v28] The grid per side that *consumers* see, as opposed to ``grid_hw``.

        ``grid_hw`` is the encoder's own grid, which only this tower's ``forward`` uses. Everything
        outside sizes itself off the reported ``patch_size`` instead -- the dataset's
        ``<|image_pad|>`` expansion and Qwen's ``grid.prod() // spatial_merge_size**2`` feature
        split both compute ``(image_size // patch_size)**2 // spatial_merge_size**2``. The two
        grids coincide for the default merger and diverge exactly when a ``merger=`` flag changes
        the spatial reduction: ``ps1`` emits ``grid_hw**2`` tokens, not ``(grid_hw / 2)**2``.

        **Any grid_thw handed to Qwen must be built from this, never from ``grid_hw``.** Getting
        that wrong does not fail loudly at construction; it surfaces as a `split_with_sizes`
        mismatch deep inside the first forward pass that takes the path in question.
        """
        return self.image_size // self.patch_size

    def make_dummy_inputs(self):
        """Zero image + grid for the no-image DeepSpeed path (keeps the vision graph connected)."""
        dummy_pixel = torch.zeros(
            (1, 3, self.image_size, self.image_size), device=self.device, dtype=self.dtype
        )
        side = self.reported_grid_hw          # [v28] not grid_hw; see that property
        dummy_grid = torch.tensor([[1, side, side]], device=self.device)
        return dummy_pixel, dummy_grid

    def _prepare(self, pixel_values: torch.Tensor, name: str) -> torch.Tensor:
        """Resize the raw [0, 1] batch to one encoder's input size and apply its normalisation."""
        size = self.input_sizes[name]
        x = pixel_values
        if x.shape[-1] != size:
            # Antialiased resampling of the *pixels* matters (these are natural images going into
            # a pretrained stem), but torch's antialias kernel has no bf16 implementation, so do it
            # in fp32 and cast back. Only ~one [n,3,512,512] temporary.
            x = F.interpolate(
                x.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True
            ).to(pixel_values.dtype)
        mean = getattr(self, f"{name}_mean").to(dtype=x.dtype)
        std = getattr(self, f"{name}_std").to(dtype=x.dtype)
        return (x - mean) / std

    def _encode_lemonfm(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            hidden = x
            for stage in self.lemonfm.features:
                hidden = checkpoint(stage, hidden, use_reentrant=False)
        else:
            hidden = self.lemonfm.features(x)
        return self.lemonfm.classifier[0](hidden)  # trained LayerNorm2d, applied per position

    def _feature_map(self, pixel_values: torch.Tensor, name: str) -> torch.Tensor:
        """One encoder's ``[n, C, grid_hw, grid_hw]`` feature map."""
        x = self._prepare(pixel_values, name)
        n = x.shape[0]

        if name == "siglip2":
            hidden = self.siglip(pixel_values=x).last_hidden_state  # [n, gs**2, 1152]
            gs = int(hidden.shape[1] ** 0.5)
            feat = hidden.view(n, gs, gs, -1).permute(0, 3, 1, 2)
        elif name == "lemonfm":
            feat = self._encode_lemonfm(x)                          # [n, 1536, S/32, S/32]
        elif name == "internvideo3":
            flat, grid_thw = patchify(x)
            hidden, _ = self.internvideo3(flat, grid_thw)           # [n*g**2, 1152], merger=Identity
            feat = to_spatial_map(hidden, n, x.shape[-1] // INTERNVIDEO3_PATCH)
        elif name == "gsvit":
            feat = gsvit_feature_map(self.gsvit, x)                 # [n, 384, S/64, S/64]
        elif name == "surgenetxl":
            feat = surgenet_feature_map(self.surgenetxl, x)         # [n, 512, S/32, S/32]
        elif name in _SURGMOTION:
            feat = surgmotion_feature_map(getattr(self, name), x)   # [n, C, S/16, S/16]
        else:
            raise ValueError(f"No feature-map path registered for encoder {name!r}.")

        if feat.shape[-1] > self.grid_hw:
            r = feat.shape[-1] // self.grid_hw
            if self.pixelshuffle and feat.shape[-1] % self.grid_hw == 0:
                # Space-to-depth (e.g. SigLIP2's 32x32x1152 -> 16x16x4608): keep every patch and let
                # the merger weigh them, instead of averaging three quarters of them away.
                feat = F.pixel_unshuffle(feat, r)
            else:
                # Downscale (e.g. SigLIP2's 32x32 -> 16x16): area-average rather than bilinear, which
                # would alias by dropping three quarters of the patches' contribution.
                feat = F.adaptive_avg_pool2d(feat, (self.grid_hw, self.grid_hw))
        elif feat.shape[-1] < self.grid_hw:
            feat = F.interpolate(
                feat, size=(self.grid_hw, self.grid_hw), mode="bilinear", align_corners=False
            )
        # The merger was sized from `_ENCODER_STRIDE`; if a backbone's real map disagrees, say so
        # here rather than letting a wrongly-shaped concat reach the merger.
        expected = self.stream_dims[self.encoder_names.index(name)]
        if feat.shape[1] != expected:
            raise RuntimeError(
                f"{name}: feature map has {feat.shape[1]} channels but the merger was built for "
                f"{expected} (shuffle factor {self.shuffle_factors[name]}); check _ENCODER_STRIDE."
            )
        return feat

    def forward(self, pixel_values: torch.Tensor, grid_thw=None, return_dict: bool = True, **kwargs):
        # pixel_values: [n, 3, S, S] raw [0, 1] (one row per image, or per video frame).
        pixel_values = pixel_values.to(self.dtype)
        maps = [self._feature_map(pixel_values, name) for name in self.encoder_names]
        # Flatten to tokens per stream before fusing: with `prenorm` the merger normalises each
        # stream over its own channels first, otherwise this is a plain channel concat.
        streams = [feat.flatten(2).transpose(1, 2) for feat in maps]   # each [n, grid_hw**2, C_i]
        if self.prenorm:
            hidden = self.merger.fuse(streams)           # [n, grid_hw**2, sum(C_i)]
        else:
            hidden = torch.cat(streams, dim=-1)          # [n, grid_hw**2, sum(C_i)]
        merged = self.merger(hidden, self.grid_hw)       # [n, (grid_hw/m)**2, out_hidden_size]
        pooled = merged.reshape(-1, merged.shape[-1])    # [n*tokens_per_image, out_hidden_size]
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden,
            pooler_output=pooled,
            deepstack_features=[],
        )


def build_multi_vision_tower(qwen_config, encoder_id: str | None = None):
    """Construct a fused multi-encoder tower sized to a loaded Qwen3-VL config."""
    encoder_id = encoder_id or "multi"
    vision_config = getattr(qwen_config, "vision_config", None)
    out_hidden_size = getattr(vision_config, "out_hidden_size", None)
    if out_hidden_size is None:
        text_config = getattr(qwen_config, "text_config", qwen_config)
        out_hidden_size = text_config.hidden_size
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2) if vision_config else 2
    return MultiVisionTowerWithMerger(
        encoder_id=encoder_id,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
    )
