#!/usr/bin/env python
"""Prove a trained stage directory loads and runs on ONE GPU.

Training runs data-parallel across both H100s, but the deliverable is a checkpoint that serves on a
single card. ZeRO-2 keeps a full parameter replica per rank, so `trainer.save_model` should write
complete safetensors shards -- but "should" is what this script replaces with evidence.

Run it with a single GPU visible; that is the point. This script loads with device_map="auto", which
would happily *split* a model across two visible cards and mask the very failure we are checking
for, so restricting the visible set is what makes the check mean anything:

    CUDA_VISIBLE_DEVICES=0 python scripts/check_single_gpu_load.py output/v13_2gpu/s1_merger

Checks, in order:
  1. the directory holds real weight shards (not a ZeRO shard dump, not an empty dir),
  2. every parameter lands on one and the same CUDA device,
  3. the model produces tokens.

Note: a stage-3 LoRA output dir (adapter + non_lora_state_dict.bin) is not loadable on its own --
check `s3_merged` instead, which is what merge_lora_weights.py produces.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--skip-generate", action="store_true",
                    help="Only check placement, do not run a forward pass.")
    args = ap.parse_args()

    path = os.path.abspath(args.model_path)
    print(f"[1gpu-check] {path}")

    if not torch.cuda.is_available():
        sys.exit("[1gpu-check] FAIL: no CUDA device visible")
    visible = torch.cuda.device_count()
    if visible != 1:
        print(f"[1gpu-check] WARNING: {visible} GPUs visible. This check is only meaningful with "
              f"one -- re-run under CUDA_VISIBLE_DEVICES=0.")

    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        sys.exit(f"[1gpu-check] FAIL: no *.safetensors in {path}. If this is a LoRA stage dir, "
                 f"check the merged output instead.")
    total_gb = sum(os.path.getsize(s) for s in shards) / 2**30
    print(f"[1gpu-check] {len(shards)} shard(s), {total_gb:.2f} GiB on disk")

    from utils import load_pretrained_model, get_model_name_from_path

    torch.cuda.reset_peak_memory_stats()
    processor, model = load_pretrained_model(
        model_path=path,
        model_base=None,
        model_name=get_model_name_from_path(path),
        device_map="auto",                       # same call shape the eval script uses
        torch_dtype=getattr(torch, args.dtype),
        use_flash_attn=True,
    )
    model.eval()

    devices = {str(p.device) for p in model.parameters()}
    buffers = {str(b.device) for b in model.buffers()}
    placement = devices | buffers
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[1gpu-check] {n_params/1e9:.2f}B params on device(s): {sorted(placement)}")

    offloaded = {d for d in placement if not d.startswith("cuda")}
    if offloaded:
        sys.exit(f"[1gpu-check] FAIL: parameters live off-GPU ({sorted(offloaded)}) -- the model "
                 f"did not fit and was offloaded to CPU/disk.")
    if len(placement) != 1:
        sys.exit(f"[1gpu-check] FAIL: model is split across {sorted(placement)} -- not "
                 f"single-GPU loadable.")

    if not args.skip_generate:
        messages = [{"role": "user", "content": [{"type": "text", "text": "Reply with the word OK."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0].strip()
        print(f"[1gpu-check] generate() returned: {reply!r}")

    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[1gpu-check] peak {peak:.2f} GiB of {total:.1f} GiB on "
          f"{torch.cuda.get_device_name(0)}")
    print("[1gpu-check] OK -- loads and runs on a single GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
