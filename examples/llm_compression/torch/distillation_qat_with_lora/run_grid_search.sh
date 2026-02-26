#!/bin/bash
set -euo pipefail

PRETRAINED="meta-llama/Llama-3.2-1B-Instruct"
BASE_OUTPUT="grid_search_results"
LOG_FILE="grid_search.log"

# ── Hyperparameter grid ─────────────────────────────────────────────
FQ_LRS=(0 1e-3 1e-4 1e-5)
LORA_LRS=(0 1e-3 1e-4 1e-5)

# Each row: warmup_epochs  constant_epochs  cosine_epochs
SCHEDULES=(
    "0   0  15"
    "0  15   0"
    "1   1  13"
    "0   1  14"
    "1   5  10"
)

# ── Helper: run lm_eval with lambada_openai ─────────────────────────
run_lm_eval() {
    local model_dir="$1"
    local log="$2"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks lambada_openai \
        --output_path "${model_dir}/lm_eval_results" \
        --batch_size auto >> "$log" 2>&1
}

# ── Main grid loop ──────────────────────────────────────────────────
total=0
for fq_lr in "${FQ_LRS[@]}"; do
for lora_lr in "${LORA_LRS[@]}"; do
for sched in "${SCHEDULES[@]}"; do
    # Skip the case where both LRs are 0 (nothing to train)
    if [[ "$fq_lr" == "0" && "$lora_lr" == "0" ]]; then
        continue
    fi

    read -r warmup constant cosine <<< "$sched"
    total=$((total + 1))

    RUN_NAME="fq${fq_lr}_lora${lora_lr}_w${warmup}_c${constant}_cos${cosine}"
    OUTPUT_DIR="${BASE_OUTPUT}/${RUN_NAME}"
    STRIPPED_DIR="${OUTPUT_DIR}/last/stripped"

    echo "============================================================"
    echo "[${total}] ${RUN_NAME}"
    echo "  fq_lr=${fq_lr}  lora_lr=${lora_lr}  warmup=${warmup}  constant=${constant}  cosine=${cosine}"
    echo "  output_dir=$(realpath -m "$OUTPUT_DIR")"
    echo "============================================================"

    # ── Training ────────────────────────────────────────────────
    python main.py \
        --pretrained "$PRETRAINED" \
        --fq_lr "$fq_lr" \
        --lora_lr "$lora_lr" \
        --warmup_epochs "$warmup" \
        --constant_epochs "$constant" \
        --cosine_epochs "$cosine" \
        --min_lr_ratio 0.1 \
        --output_dir "$OUTPUT_DIR" \
        >> "$LOG_FILE" 2>&1

    # ── Evaluation ──────────────────────────────────────────────
    echo "  Running lm-eval (lambada_openai) …"
    run_lm_eval "$STRIPPED_DIR" "$LOG_FILE"

done
done
done

echo ""
echo "Grid search complete — ${total} runs.  Results in ${BASE_OUTPUT}/"
echo "View MLflow UI:  mlflow ui --backend-store-uri ${BASE_OUTPUT}/*/mlruns"
