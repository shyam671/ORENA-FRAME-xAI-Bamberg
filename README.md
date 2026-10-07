# ORENA-FRAME-xAI-Bamberg

A self-contained training run for Qwen3-VL-4B-Instruct with a fused **SigLIP2-SO400M + LemonFM** vision encoder, fine-tuned on FOCUS surgical VQA frames with a 3-stage curriculum.


## What the run does

The vision encoder string is:

```
multi:siglip2=google/siglip2-so400m-patch16-512+lemonfm=<LEMONFM>@16
```

Both encoders' feature maps are resampled to a 16×16 grid and concatenated channel-wise. A freshly initialised pixel-shuffle projector (the **merger**) maps the result to the LLM hidden size. The LLM is frozen in every stage; stage 3 adapts it through LoRA.

| Stage | Trains | Epochs | merger LR | vision LR | Seed | per-device batch |
|---|---|---|---|---|---|---|
| `s1_merger` | merger | 1 | 1e-4 | – | 42 | 16 |
| `s2_merger_vision` | merger + both encoders | 2 | 5e-5 | 2e-6 | 43 | 8 |
| `s3_merger_llm_lora` | merger + LLM LoRA (r=32, α=64) | 4 | 2e-5 | – | 44 | 8 |

Settings shared by all stages:
- Global batch is 32. Gradient accumulation is derived from `NGPU`.
- AdamW with weight decay 0.1, a cosine schedule and 3% warmup. The LLM/LoRA learning rate is 1e-4.
- bf16, DeepSpeed ZeRO-2, Liger kernels, FlashAttention-2 and gradient checkpointing.
- A curriculum sampler (`scripts/curriculum_schedule.json`) stretches from 0.25 to 1.00 over each stage's epochs.
- Each stage starts from the previous stage's output directory. After stage 3, the LoRA adapter is merged onto stage 2 to produce `s3_merged`.

Evaluation and checkpointing happen once per stage, at its last step. `eval_loss` is measured on `val_all_new.json`, which holds 4 videos that do not appear in the training split.

## Requirements

- Conda env with torch 2.8, torchvision 0.23, transformers 5.x, deepspeed, peft, trl, liger-kernel, flash-attn, qwen-vl-utils, ujson and pandas. Both scripts activate this env automatically.
- **CUDA 12.9 toolkit** at `/usr/local/cuda-12.9`, which DeepSpeed uses to JIT-build `cpu_adam`. To use a different toolkit, set `CUDA_HOME_OVERRIDE`.
- **Inputs**, each with an env var to override the default path:

| Variable | Default | What it is |
|---|---|---|
| `LEMONFM` | `/data/local/hf/hub/pretrained-vision-encoder/lemonfm.pth` | LemonFM ConvNeXt weights |
| `DATA` | `/home/staff/srai/orena/data/focus_vqa_frame/frame` | Directory holding the splits |
| `TRAIN_JSON` | `$DATA/train_all_new.json` | Training split |
| `VAL_JSON` | `$DATA/val_all_new.json` | Held-out split (4 videos) |
| `QTYPE_MAP` | `$DATA/qtype_map_all.json` | Question-type sidecar for the curriculum. It is built automatically if missing. |
| `FRAMES` | `/data/local/orena/dataset` | Parquet root, also used as the image folder |
| `OUT` | `./output/v13_2gpu` | Output directory |

The base model `Qwen/Qwen3-VL-4B-Instruct` and the SigLIP2 encoder are downloaded from the Hugging Face Hub, or read from your HF cache.

## Usage

```bash
bash run_v13.sh                   # the full run
```

## Outputs

```
output/
├── s1_merger/              loadable model
├── s2_merger_vision/       loadable model
├── s3_merger_llm_lora/     LoRA adapter + non_lora_state_dict.bin (NOT loadable alone)
└── s3_merged/              stage 3, merged onto s2 — the deliverable
```

## Folder map

| Path | Role |
|---|---|
| `run_v13.sh` | Runs the whole pipeline: pre-flight, qtype sidecar, the 3 training stages, then the LoRA merge |
| `src/` | A full copy of the fine-tuning code. `train/train_sft.py` is the entry point, `model/multi_vision_tower.py` builds the fusion encoder, `dataset/curriculum_sampler.py` is the curriculum, and `merge_lora_weights.py` does the merge. |
| `scripts/zero2.json` | DeepSpeed ZeRO-2 config |
| `scripts/curriculum_schedule.json` | Per-question-type curriculum schedule |
| `scripts/build_qtype_labels.py` | Builds the qtype sidecar from the training split and parquet metadata |
| `scripts/check_splits.py` | Checks that the sidecar covers every training id and that train and val share no ids |
| `scripts/check_curriculum_sharding.py` | Checks that the curriculum batches are unchanged when sharded across ranks |
| `scripts/check_single_gpu_load.py` | Loads a saved model on one GPU and runs a short generation |

## Acknowledgement

This project is based on github.com/2U1/Qwen-VL-Series-Finetune
