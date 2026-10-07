"""Generation-time SigLIP2 preprocessing for a vision-swapped Qwen3-VL model.

Mirrors the training dataset's SigLIP path (fixed square images + manual <|image_pad|>
expansion + synthesized grid_thw) but for batched `model.generate`: left-padded, no labels,
`add_generation_prompt=True`. Kept separate from the training dataset so the eval script can
import just this without the Trainer/DataArguments machinery.
"""

from __future__ import annotations

import torch

IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"


class SiglipGenerationPreprocessor:
    def __init__(self, tokenizer, siglip_image_processor, qwen_config, siglip_vision_config):
        self.tok = tokenizer
        self.ip = siglip_image_processor
        merge = getattr(qwen_config.vision_config, "spatial_merge_size", 2)
        self.grid_hw = siglip_vision_config.image_size // siglip_vision_config.patch_size
        merged_hw = self.grid_hw // merge
        self.tokens_per_image = merged_hw * merged_hw
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.video_token_id = tokenizer.convert_tokens_to_ids(VIDEO_TOKEN)
        self.pad_id = tokenizer.pad_token_id
        # Mirrors training: feed each image as a 4-view rotation stack when the checkpoint
        # was trained that way (config.rotate_augment, persisted by train_sft).
        self.rotate_augment = getattr(qwen_config, "rotate_augment", False)

    def image_pixels(self, image):
        """SigLIP pixel_values [1,3,S,S] for a PIL image."""
        return self.ip(images=image, return_tensors="pt")["pixel_values"]

    def video_pixels(self, frames):
        """SigLIP pixel_values [T,3,S,S] for a list/tensor of frames; returns (pixels, T)."""
        px = self.ip(images=frames, return_tensors="pt")["pixel_values"]
        return px, px.shape[0]

    def _expand(self, ids, frame_counts):
        exp_ids, exp_types = [], []
        ptr = 0
        for t in ids:
            if t == self.image_token_id or t == self.video_token_id:
                n = self.tokens_per_image * frame_counts[ptr]
                ptr += 1
                exp_ids.extend([t] * n)
                exp_types.extend([1 if t == self.image_token_id else 2] * n)
            else:
                exp_ids.append(t)
                exp_types.append(0)
        return exp_ids, exp_types

    def build(self, texts, per_sample_media):
        """Build a left-padded generation batch.

        texts: list[str] chat-templated prompts (with single <|image_pad|>/<|video_pad|> each).
        per_sample_media: list (one per sample) of list of dicts:
            {"pixels": tensor[rows,3,S,S], "frames": int, "is_video": bool} in appearance order.
        Returns a dict of tensors ready for `model.generate`.
        """
        seqs, types = [], []
        image_pixels, image_grids = [], []
        video_pixels, video_grids = [], []

        for text, media in zip(texts, per_sample_media):
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            frame_counts = [m["frames"] for m in media]
            exp_ids, exp_types = self._expand(ids, frame_counts)
            seqs.append(torch.tensor(exp_ids, dtype=torch.long))
            types.append(torch.tensor(exp_types, dtype=torch.long))
            for m in media:
                grid = torch.tensor([[m["frames"], self.grid_hw, self.grid_hw]])
                if m["is_video"]:
                    video_pixels.append(m["pixels"])
                    video_grids.append(grid)
                else:
                    image_pixels.append(m["pixels"])
                    image_grids.append(grid)

        max_len = max(s.size(0) for s in seqs)
        batch_ids, batch_mask, batch_types = [], [], []
        for s, t in zip(seqs, types):
            pad = max_len - s.size(0)
            pad_ids = torch.full((pad,), self.pad_id, dtype=torch.long)
            batch_ids.append(torch.cat([pad_ids, s]))                       # left pad
            batch_mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones_like(s)]))
            batch_types.append(torch.cat([torch.zeros(pad, dtype=torch.long), t]))

        out = {
            "input_ids": torch.stack(batch_ids),
            "attention_mask": torch.stack(batch_mask),
            "mm_token_type_ids": torch.stack(batch_types),
        }
        if image_pixels:
            out["pixel_values"] = torch.cat(image_pixels, dim=0)
            out["image_grid_thw"] = torch.cat(image_grids, dim=0)
        if video_pixels:
            out["pixel_values_videos"] = torch.cat(video_pixels, dim=0)
            out["video_grid_thw"] = torch.cat(video_grids, dim=0)
        return out


def load_siglip_generation_model(
    model_id, device, processor, *, dtype=None, attn_implementation=None
):
    """Load a SigLIP2-swapped Qwen3-VL checkpoint for generation.

    Returns ``(model, preprocessor, vision_encoder_id)``. If ``model_id`` is not a
    SigLIP-swapped checkpoint (no ``vision_encoder_id`` in its config), returns
    ``(None, None, None)`` so the caller can fall back to the native Qwen load path.

    ``dtype`` and ``attn_implementation`` are forwarded to ``from_pretrained``. Both default
    to ``None``, which reproduces the historical behaviour exactly: fp16 on CUDA (fp32 on CPU)
    and whatever attention backend transformers picks (SDPA). They exist so an ablation can
    load the same checkpoint in bf16, or with flash-attention-2, without editing this file.
    Note that the swapped tower is built separately by ``build_vision_tower`` and takes no
    attention kwarg, so ``attn_implementation`` reaches the text decoder only; the tower does
    pick up ``dtype``, since ``maybe_swap_vision_encoder`` casts it to the embedding dtype.
    """
    from transformers import AutoConfig
    from model.load_model import load_qwen_vl_generation_model
    from model.vision_encoder import load_vision_encoder_assets

    cfg = AutoConfig.from_pretrained(model_id)
    encoder_id = getattr(cfg, "vision_encoder_id", None)
    if not encoder_id:
        return None, None, None

    use_cuda = str(device).startswith("cuda")
    load_kwargs = {
        "dtype": dtype or (torch.float16 if use_cuda else torch.float32),
        "device_map": "auto" if use_cuda else None,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = load_qwen_vl_generation_model(model_id, **load_kwargs).eval()
    siglip_ip, siglip_vcfg = load_vision_encoder_assets(encoder_id)
    prep = SiglipGenerationPreprocessor(processor.tokenizer, siglip_ip, cfg, siglip_vcfg)
    return model, prep, encoder_id


def _extract_media(message_media, prep):
    """SigLIP pixels + frame count for one rendered media message entry."""
    if message_media["type"] == "image":
        img = message_media["image"]
        if getattr(prep, "rotate_augment", False):
            from dataset.data_utils import rotate_views

            views = rotate_views(img)
            px = prep.image_pixels(views)  # [len(views), 3, S, S]
            return {"pixels": px, "frames": px.shape[0], "is_video": False}
        return {"pixels": prep.image_pixels(img), "frames": 1, "is_video": False}
    from qwen_vl_utils import fetch_video

    ele = {"type": "video", "video": message_media["video"]}
    if "fps" in message_media:
        ele["fps"] = message_media["fps"]
    frames = fetch_video(ele)
    images = list(frames) if isinstance(frames, (list, tuple)) else frames
    pixels, num_frames = prep.video_pixels(images)
    return {"pixels": pixels, "frames": num_frames, "is_video": True}


def siglip_prepare(processor, prep, messages_list):
    """The CPU half of :func:`siglip_generate` -- chat templating + SigLIP pixel prep.

    Split out so a caller can run it on a worker thread for batch *k+1* while the GPU is still
    busy with batch *k* (`prep.image_pixels` resizes every frame to a 512x512 fp32 tensor, which
    is not free). Returns ``(texts, per_sample_media)``, exactly what ``prep.build`` consumes, so
    routing through here is numerically identical to letting `siglip_generate` do it inline.
    """
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    per_sample_media = [[_extract_media(m[1]["content"][0], prep)] for m in messages_list]
    return texts, per_sample_media


def siglip_generate(
    model,
    processor,
    prep,
    messages_list,
    device,
    max_new_tokens=128,
    prepared=None,
    greedy=False,
    **gen_overrides,
):
    """Batched generation for a SigLIP2-swapped model. Returns one decoded string per sample.

    ``messages_list`` is a list of chat-message lists whose user turn's first content entry is
    the media dict ({"type": "image"/"video", ...}) -- exactly what the eval engine already builds.

    ``prepared`` is an optional pre-computed ``(texts, per_sample_media)`` from
    :func:`siglip_prepare`; when omitted it is computed here, so the default path is unchanged.
    ``greedy`` forces ``do_sample=False``, overriding the checkpoint's generation_config (which
    ships ``do_sample=True, temperature=0.7, top_k=20`` -- i.e. eval is otherwise
    non-deterministic run to run).

    ``gen_overrides`` are passed straight through to ``model.generate`` and take precedence over
    both of the above, so a caller can set ``temperature``/``top_k``/``top_p`` instead of
    silently inheriting the checkpoint's generation_config. Without this, sweeping those
    parameters is a no-op on the SigLIP path: this function used to build ``gen_kwargs`` from
    scratch and drop anything the caller asked for.
    """
    texts, per_sample_media = (
        prepared if prepared is not None else siglip_prepare(processor, prep, messages_list)
    )
    batch = prep.build(texts, per_sample_media)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if greedy:
        gen_kwargs["do_sample"] = False
    gen_kwargs.update(gen_overrides)
    with torch.no_grad():
        gen_ids = model.generate(**batch, **gen_kwargs)
    trimmed = gen_ids[:, batch["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)
