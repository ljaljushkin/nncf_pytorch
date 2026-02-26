#!/bin/bash
set -euo pipefail

PRETRAINED="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="output_llama_1b"
LOG_FILE="grid_search.log"

# ── Hyperparameter grid ─────────────────────────────────────────────
FQ_LRS=(0 1e-3 1e-5)
LORA_LRS=(0 1e-3 1e-5)
FQ_WEIGHT_DECAYS=(0 1e-3)
LORA_WEIGHT_DECAYS=(1e-4)

# # Each row: warmup_epochs  constant_epochs  cosine_epochs
SCHEDULES=(
    "0   0  15"
    # "0  15   0"
    "1   1  13"
    # "0   1  14"
    # "1   5  9"
)

# ── TEST ──────────────────────────────────────────────────
# FQ_LRS=(0)
# LORA_LRS=(1)
# FQ_WEIGHT_DECAYS=(0)
# LORA_WEIGHT_DECAYS=(1e-3)

# # Each row: warmup_epochs  constant_epochs  cosine_epochs
# SCHEDULES=(
#     "0  0   1"
# )

# ── Main grid loop ──────────────────────────────────────────────────
# Pre-compute total number of configurations (excluding both-LRs-zero).
total_configs=0
for fq_lr in "${FQ_LRS[@]}"; do
for lora_lr in "${LORA_LRS[@]}"; do
    if [[ "$fq_lr" == "0" && "$lora_lr" == "0" ]]; then continue; fi
    total_configs=$(( total_configs + ${#FQ_WEIGHT_DECAYS[@]} * ${#LORA_WEIGHT_DECAYS[@]} * ${#SCHEDULES[@]} ))
done
done
echo "Total configurations: ${total_configs}"

current=0
for fq_lr in "${FQ_LRS[@]}"; do
for lora_lr in "${LORA_LRS[@]}"; do
for fq_wd in "${FQ_WEIGHT_DECAYS[@]}"; do
for lora_wd in "${LORA_WEIGHT_DECAYS[@]}"; do
for sched in "${SCHEDULES[@]}"; do
    # Skip the case where both LRs are 0 (nothing to train)
    if [[ "$fq_lr" == "0" && "$lora_lr" == "0" ]]; then
        continue
    fi

    read -r warmup constant cosine <<< "$sched"
    current=$((current + 1))

    RUN_NAME="fq${fq_lr}_lora${lora_lr}_fqwd${fq_wd}_lorawd${lora_wd}_w${warmup}_c${constant}_cos${cosine}"

    echo "============================================================"
    echo "[${current}/${total_configs}] ${RUN_NAME}"
    echo "  fq_lr=${fq_lr}  lora_lr=${lora_lr}  fq_wd=${fq_wd}  lora_wd=${lora_wd}  warmup=${warmup}  constant=${constant}  cosine=${cosine}"
    echo "  output_dir=$(realpath -m "$OUTPUT_DIR")"
    echo "============================================================"

    # ── Training ────────────────────────────────────────────────
    python main.py \
        --pretrained "$PRETRAINED" \
        --fq_lr "$fq_lr" \
        --lora_lr "$lora_lr" \
        --fq_weight_decay "$fq_wd" \
        --lora_weight_decay "$lora_wd" \
        --warmup_epochs "$warmup" \
        --constant_epochs "$constant" \
        --cosine_epochs "$cosine" \
        --min_lr_ratio 0.1 \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --resume \
        >> "$LOG_FILE" 2>&1

done
done
done
done
done

echo ""
echo "Grid search complete -- ${current}/${total_configs} runs."
echo "View MLflow UI:  mlflow ui --backend-store-uri sqlite:////${OUTPUT_DIR}/mlflow.db"
