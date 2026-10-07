#!/bin/bash
# v13 -- SigLIP2-SO400M + LemonFM concat fusion, 3-stage curriculum, trained on the new all-splits
# resplit (train_all_new.json / val_all_new.json).
# Standalone copy of Qwen-VL-Series-Finetune-master/scripts/arxiv_do_not_touch_these_files/
# finetune_focus_fusion_v13.sh; same run, minus the eval phase.
#
#   s1  merger only              1 epoch
#   s2  merger + both encoders   2 epochs
#   s3  merger + LLM via LoRA    4 epochs  ->  merged
#
# Both feature maps are resampled to a 16x16 grid and concatenated channel-wise; one fresh
# pixel-shuffle projector (the "merger") maps them to the LLM hidden size. Each stage resumes the
# previous stage's output_dir -- the saved config.vision_encoder_id rebuilds the same fusion.
# The LLM is frozen throughout; stage 3 reaches it through LoRA because train_sft.py rejects
# lora_enable together with an unfrozen LLM, and a full 4B fine-tune does not fit 48 GB.
#
# One curriculum sweep per stage: the sampler stretches 0.25->1.00 over the stage's whole epoch
# budget, so each stage is a single process with its real --num_train_epochs. Eval and checkpoint
# fire once per stage, at its final optimizer step.
#
# The splits are the pre-built ones from split_manifest_new.json: val_all_new.json is every
# non-Temporal QA of 4 held-out videos and those videos appear nowhere in train_all_new.json, so
# eval_loss is a genuine held-out number.
#
# Benchmark evaluation is not part of this script; see README.md ("Evaluating") for how to score
# s1_merger / s2_merger_vision / s3_merged with orena/eval/inference-combined-batched.py.
#
# Env knobs: NGPU, LEMONFM, DATA, TRAIN_JSON, VAL_JSON, QTYPE_MAP, FRAMES, OUT, COOLDOWN,
# CUDA_HOME_OVERRIDE, REBUILD_DATA=1 rebuild the qtype sidecar, RESUME=1 continue a dirty output
# dir, PROBE=1 measure memory/throughput for 12 steps per stage.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# Both cards: Accelerate shards the sampler's global stream round-robin, so each optimizer step
# still sees the exact curriculum batch -- verified by scripts/check_curriculum_sharding.py.
NGPU="${NGPU:-2}"
# NGPU=1 must also hide the second card, or device_map="auto" downstream would still straddle both.
if [ "$NGPU" -eq 1 ]; then export CUDA_VISIBLE_DEVICES=0; else export CUDA_VISIBLE_DEVICES=0,1; fi
# DeepSpeed JIT-builds cpu_adam against CUDA_HOME and refuses a major-version mismatch;
# torch is built for CUDA 12.8 and the system default is 13.3, so point at 12.9.
export CUDA_HOME="${CUDA_HOME_OVERRIDE:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0          # H100 NVL only (SM 9.0); 8.9 builds no usable cubin here
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation at the larger batch
export OMP_NUM_THREADS=8                 # 64 cores / 2 ranks; stops dataloader CPU oversubscription

# torch 2.8 + torchvision 0.23, needed by the LemonFM ConvNeXt.
if [ "${CONDA_DEFAULT_ENV:-}" != "qwen-vl-finetune" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qwen-vl-finetune
fi

BASE_MODEL="Qwen/Qwen3-VL-4B-Instruct"
LEMONFM="${LEMONFM:-/data/local/hf/hub/pretrained-vision-encoder/lemonfm.pth}"
ENCODER="multi:siglip2=google/siglip2-so400m-patch16-512+lemonfm=$LEMONFM@16"

DATA="${DATA:-/home/staff/srai/orena/data/focus_vqa_frame/frame}"
TRAIN_JSON="${TRAIN_JSON:-$DATA/train_all_new.json}"
VAL_JSON="${VAL_JSON:-$DATA/val_all_new.json}"
QTYPE_MAP="${QTYPE_MAP:-$DATA/qtype_map_all.json}"
FRAMES="${FRAMES:-/data/local/orena/dataset}"   # parquet root, doubles as the image folder
OUT="${OUT:-$REPO/output/v13_2gpu}"

GLOBAL_BATCH=32                          # doubled from 16; two ranks now fill it without accumulating
COOLDOWN="${COOLDOWN:-1}"              # seconds between stages, to let the GPU settle

# The LLM stays frozen and the merger stays trainable in every stage; the LLM/LoRA rate is 1e-4.
# Learning rates are deliberately UNCHANGED from the 16-batch run -- do not "fix" them into a rescale.
# batch is per device (accum is derived); gc=gradient checkpointing, kept on until PROBE says otherwise.
#        name                epochs  lora   freeze_tower  merger_lr  vision_lr  seed  batch  gc
STAGES=(
    "s1_merger               1       False  True          1e-4       -          42    16     True"
    "s2_merger_vision        2       False  False         5e-5       2e-6       43    8      True"
    "s3_merger_llm_lora      4       True   True          2e-5       -          44    8      True"
)

banner() {
    echo "=============================================================="
    printf '%s\n' "$@"
    echo "=============================================================="
}

# ── Pre-flight ─────────────────────────────────────────────────────────
# The whole point of this tower is LemonFM's surgical pretraining; v9/v10 shipped a path that
# silently did not exist, so fail here rather than 20 minutes into stage 1.
if [ ! -f "$LEMONFM" ]; then
    echo "[ERROR] LemonFM weights not found: $LEMONFM (set LEMONFM)" >&2
    exit 1
fi
for f in "$TRAIN_JSON" "$VAL_JSON"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] missing data split: $f" >&2
        exit 1
    fi
done

# ── Phase 0: qtype sidecar for the training file ───────────────────────
# The splits themselves are pre-built and never touched here.
# The stock qtype_map.json only covers the old train.json ids. Unlisted ids fall into
# "__unknown__", which would flatten the curriculum for most of the set, so use a sidecar
# spanning the whole training file.
if [ "${REBUILD_DATA:-0}" = "1" ] || [ ! -f "$QTYPE_MAP" ]; then
    banner "[DATA] building qtype sidecar -> $QTYPE_MAP"
    python "$REPO/scripts/build_qtype_labels.py" \
        --data_path "$TRAIN_JSON" \
        --parquet_root "$FRAMES" \
        --out "$QTYPE_MAP"
else
    echo "[DATA] reusing $QTYPE_MAP (REBUILD_DATA=1 to rebuild)"
fi

# Sidecar coverage + train/val id-disjointness.
python "$REPO/scripts/check_splits.py" "$TRAIN_JSON" "$VAL_JSON" "$QTYPE_MAP"

# Step budgets follow the data instead of a hard-coded number; matches CurriculumSampler.__len__.
NUM_TRAIN=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$TRAIN_JSON")
STEPS_PER_EPOCH=$(( (NUM_TRAIN + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
echo "[DATA] $NUM_TRAIN records -> $STEPS_PER_EPOCH steps/epoch (global batch $GLOBAL_BATCH)"

if [ "$NGPU" -gt 1 ]; then
    # An Accelerate change to the sharding pattern would reshape every step's category mix silently
    # rather than erroring, so assert the curriculum survives 2 ranks before committing 7 epochs.
    python "$REPO/scripts/check_curriculum_sharding.py" \
        --train "$TRAIN_JSON" --qtype_map "$QTYPE_MAP" \
        --global_batch "$GLOBAL_BATCH" --world "$NGPU"
fi

# ── Phase 1: staged training ───────────────────────────────────────────
for i in "${!STAGES[@]}"; do
    read -r name epochs lora freeze_tower merger_lr vision_lr seed batch gc <<<"${STAGES[$i]}"
    stage_dir="$OUT/$name"
    stage_steps=$(( STEPS_PER_EPOCH * epochs ))

    # Derived so the stage table sets only the per-device batch; a non-integral accum would
    # desynchronise the optimizer step from the sampler's fixed GLOBAL_BATCH-sized blocks.
    ACCUM=$(( GLOBAL_BATCH / (batch * NGPU) ))
    if [ $(( ACCUM * batch * NGPU )) -ne "$GLOBAL_BATCH" ] || [ "$ACCUM" -lt 1 ]; then
        echo "[ERROR] $name: batch=$batch x NGPU=$NGPU does not divide GLOBAL_BATCH=$GLOBAL_BATCH" >&2
        exit 1
    fi

    # PROBE measures peak memory / throughput per stage to lock in `batch` and `gc`; every stage
    # starts from the hub checkpoint because no stage actually saves anything in this mode.
    probe_args=()
    if [ "${PROBE:-0}" = "1" ]; then
        probe_args=(--max_steps 12 --skip_memory_metrics False)
        stage_dir="$OUT/probe/$name"
    fi

    # Stage 1 starts from the hub checkpoint, later stages continue the previous one.
    if [ "$i" -eq 0 ] || [ "${PROBE:-0}" = "1" ]; then
        model_in="$BASE_MODEL"
    else
        read -r prev _ <<<"${STAGES[$((i - 1))]}"
        model_in="$OUT/$prev"
        # The loader reads local shards with strict=False, so an empty directory would quietly
        # give us a randomly initialised tower instead of failing.
        if ! compgen -G "$model_in/*.safetensors" > /dev/null; then
            echo "[ERROR] previous stage produced no weights: $model_in/*.safetensors" >&2
            exit 1
        fi
    fi

    # train_sft.py auto-resumes from any checkpoint-* it finds in output_dir.
    if [ "${RESUME:-0}" != "1" ] && compgen -G "$stage_dir/checkpoint-*" > /dev/null; then
        echo "[ERROR] $stage_dir already holds a checkpoint; delete it or re-run with RESUME=1." >&2
        exit 1
    fi

    # Pass --vision_lr only while the tower trains, otherwise its param group is empty.
    vision_args=()
    if [ "$vision_lr" != "-" ]; then
        vision_args=(--vision_lr "$vision_lr")
    fi

    lora_args=()
    if [ "$lora" = "True" ]; then
        lora_args=(
            --lora_namespan_exclude "['lm_head', 'embed_tokens']"
            --lora_rank 32
            --lora_alpha 64
            --lora_dropout 0.05
            --num_lora_modules -1
        )
    fi

    banner "[TRAIN] stage $((i + 1))/${#STAGES[@]}  $name  epochs=$epochs  steps=$stage_steps${PROBE:+  [PROBE]}" \
           "        from  $model_in" \
           "        ->    $stage_dir" \
           "        lora=$lora  freeze_vision_tower=$freeze_tower  seed=$seed" \
           "        merger_lr=$merger_lr  vision_lr=$vision_lr" \
           "        gpus=$NGPU  batch=$batch/device x accum=$ACCUM = global $GLOBAL_BATCH  gc=$gc"

    deepspeed --num_gpus "$NGPU" "$REPO/src/train/train_sft.py" \
        --deepspeed scripts/zero2.json \
        --model_id "$model_in" \
        --vision_encoder_id "$ENCODER" \
        --output_dir "$stage_dir" \
        --data_path "$TRAIN_JSON" \
        --eval_path "$VAL_JSON" \
        --image_folder "$FRAMES" \
        --qtype_map_path "$QTYPE_MAP" \
        --curriculum_enable True \
        --curriculum_schedule scripts/curriculum_schedule.json \
        --curriculum_seed "$seed" \
        --lora_enable "$lora" \
        "${lora_args[@]}" \
        --freeze_vision_tower "$freeze_tower" \
        --freeze_merger False \
        --freeze_llm True \
        --num_train_epochs "$epochs" \
        --per_device_train_batch_size "$batch" \
        --per_device_eval_batch_size "$batch" \
        --gradient_accumulation_steps "$ACCUM" \
        --learning_rate 1e-4 \
        --merger_lr "$merger_lr" \
        "${vision_args[@]}" \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --eval_strategy steps --eval_steps "$stage_steps" \
        --save_strategy steps --save_steps "$stage_steps" --save_total_limit 1 \
        --logging_steps 1 \
        --report_to tensorboard \
        --bf16 True --fp16 False --tf32 True \
        --use_liger_kernel True \
        --disable_flash_attn2 False \
        --gradient_checkpointing "$gc" \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --fps 1 \
        --dataloader_num_workers 8 \
        --dataloader_persistent_workers True \
        --dataloader_prefetch_factor 4 \
        "${probe_args[@]}"

    # A stage's weights must land on disk before the next one reads them as --model_id.
    if [ "${PROBE:-0}" != "1" ] && [ "$lora" != "True" ]; then
        CUDA_VISIBLE_DEVICES=0 python "$REPO/scripts/check_single_gpu_load.py" "$stage_dir"
    fi

    if [ "$i" -ne $(( ${#STAGES[@]} - 1 )) ]; then
        echo "[COOLDOWN] sleeping ${COOLDOWN}s…"
        sleep "$COOLDOWN"
    fi
done

if [ "${PROBE:-0}" = "1" ]; then
    banner "[PROBE] done -- read train_mem_gpu_peaked_delta and train_samples_per_second above." \
           "Raise each stage's batch while peak stays under ~85 GB, then retry gc=False at that" \
           "batch and keep it only if it both fits and is measurably faster. rm -rf $OUT/probe"
    exit 0
fi

# ── Phase 2: merge the stage-3 LoRA adapter ────────────────────────────
# --model-base must be the stage-2 directory, not the hub id: non_lora_state_dict.bin only holds
# stage 3's trainable params (the merger), so the fused tower has to come from stage 2's shards.
banner "[MERGE] s3_merger_llm_lora + base s2_merger_vision -> $OUT/s3_merged"
mkdir -p "$OUT/s3_merged"
# Pinned to one card from here on: the merge must not touch GPU 1.
CUDA_VISIBLE_DEVICES=0 python "$REPO/src/merge_lora_weights.py" \
    --model-path "$OUT/s3_merger_llm_lora" \
    --model-base "$OUT/s2_merger_vision" \
    --save-model-path "$OUT/s3_merged" \
    --torch-dtype bfloat16 \
    --safe-serialization

# The merged stage-3 model is the deliverable, so prove it serves on one card.
CUDA_VISIBLE_DEVICES=0 python "$REPO/scripts/check_single_gpu_load.py" "$OUT/s3_merged"

banner "Done. Loadable models:" \
       "    $OUT/s1_merger" \
       "    $OUT/s2_merger_vision" \
       "    $OUT/s3_merged          (stage 3 -- the adapter dir alone is not loadable)" \
       "Score them with orena/eval/inference-combined-batched.py; see README.md (Evaluating)."
