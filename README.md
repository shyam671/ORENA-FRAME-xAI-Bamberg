# focus-fusion-v13

A self-contained training run for **v13**: Qwen3-VL-4B-Instruct with a fused **SigLIP2-SO400M + LemonFM** vision encoder, fine-tuned on FOCUS surgical VQA frames with a 3-stage curriculum.

This folder is a standalone copy of `Qwen-VL-Series-Finetune-master/scripts/arxiv_do_not_touch_these_files/finetune_focus_fusion_v13.sh`, together with the code it runs. The training run is identical. The only difference is that benchmark evaluation was moved out of the script (see [Evaluating](#evaluating)).

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

- **Conda env `qwen-vl-finetune`** with torch 2.8, torchvision 0.23, transformers 5.x, deepspeed, peft, trl, liger-kernel, flash-attn, qwen-vl-utils, ujson and pandas. Both scripts activate this env automatically.
- **CUDA 12.9 toolkit** at `/usr/local/cuda-12.9`, which DeepSpeed uses to JIT-build `cpu_adam`. To use a different toolkit, set `CUDA_HOME_OVERRIDE`.
- **GPUs:** the default is 2× H100 (SM 9.0). Set `NGPU=1` to use one GPU.
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
bash smoke_test.sh                # ~1 min, CPU only: env, imports, inputs, splits, sampler
SMOKE_GPU=1 bash smoke_test.sh    # also a 12-step GPU probe of every stage
bash run_v13.sh                   # the full run
```

The smoke test runs six tiers and prints `[PASS]` or `[FAIL]` for each. It runs every tier even after a failure, so one pass shows every problem.

1. **static:** the run script parses and every file is present.
2. **env:** package versions, and `nvcc` exists under `CUDA_HOME`.
3. **imports (CPU):** the training, merge, fusion-tower and sampler modules import.
4. **inputs + splits:** the weights and data files exist, the qtype sidecar covers every training id, and train and val share no ids.
5. **sampler sharding:** the curriculum batches survive Accelerate's sharding across `NGPU` ranks.
6. **GPU probe** (only with `SMOKE_GPU=1`): `PROBE=1` training for 12 steps per stage, written to `output/smoke` and deleted afterwards.

Optional settings for `run_v13.sh`:

| Variable | Effect |
|---|---|
| `NGPU=1` | Train on one GPU (only GPU 0 is visible). Accumulation is adjusted so the global batch stays 32. |
| `PROBE=1` | Run 12 steps per stage, all from the hub model, and report peak memory and throughput. Use it to tune `batch` and `gc` in the stage table. |
| `RESUME=1` | Continue from an existing `checkpoint-*` in a stage's directory. Without it the script refuses to run over existing checkpoints. |
| `REBUILD_DATA=1` | Rebuild the qtype sidecar. |
| `COOLDOWN=<s>` | Pause between stages. The default is 180 s. |

## Outputs

```
output/v13_2gpu/
├── s1_merger/              loadable model
├── s2_merger_vision/       loadable model
├── s3_merger_llm_lora/     LoRA adapter + non_lora_state_dict.bin (NOT loadable alone)
└── s3_merged/              stage 3, merged onto s2 — the deliverable
```

Every loadable directory is checked by `scripts/check_single_gpu_load.py` on a single GPU. Each saved `config.json` records `vision_encoder_id`, so loading a stage rebuilds the fusion tower automatically, as long as this folder's `src/` is importable.

## Evaluating

Evaluation is not included in this folder. It uses `orena/eval/inference-combined-batched.py`, which needs the external `focus` package and an API judge. To score a stage on the 4 held-out videos:

```bash
cd /home/staff/srai/orena/eval
CUDA_VISIBLE_DEVICES=0 \
FINETUNE_SRC=/home/staff/srai/orena/focus-fusion-v13/src \
MODEL_ID=/home/staff/srai/orena/focus-fusion-v13/output/v13_2gpu/s3_merged/ \
EVAL_OUTPUT_DIR=output/v13_2gpu/s3_merged \
python inference-combined-batched.py --manifest /home/staff/srai/orena/data/focus_vqa_frame/frame/val_all_new.json
```

Repeat for `s1_merger` and `s2_merger_vision`. The summary is written to `combined_summary.csv`; read row `lapchole,overall,MEAN`.

- `FINETUNE_SRC` must point at this folder's `src/`, so the eval can rebuild the fusion tower.
- Pin the eval to one GPU. `device_map="auto"` would otherwise split the model across both cards.
- Do not score the FOCUS TEST split. `train_all_new.json` contains 6089 of its 6252 ids, so a TEST score would measure fit, not generalisation.
- Compare these numbers only from stage to stage. Earlier runs (v8, v11, v12) were scored on the TEST split and are not comparable.

## Folder map

| Path | Role |
|---|---|
| `run_v13.sh` | Runs the whole pipeline: pre-flight, qtype sidecar, the 3 training stages, then the LoRA merge |
| `smoke_test.sh` | The pre-flight checks described in [Usage](#usage) |
| `src/` | A full copy of the fine-tuning code. `train/train_sft.py` is the entry point, `model/multi_vision_tower.py` builds the fusion encoder, `dataset/curriculum_sampler.py` is the curriculum, and `merge_lora_weights.py` does the merge. |
| `scripts/zero2.json` | DeepSpeed ZeRO-2 config |
| `scripts/curriculum_schedule.json` | Per-question-type curriculum schedule |
| `scripts/build_qtype_labels.py` | Builds the qtype sidecar from the training split and parquet metadata |
| `scripts/check_splits.py` | Checks that the sidecar covers every training id and that train and val share no ids |
| `scripts/check_curriculum_sharding.py` | Checks that the curriculum batches are unchanged when sharded across ranks |
| `scripts/check_single_gpu_load.py` | Loads a saved model on one GPU and runs a short generation |
