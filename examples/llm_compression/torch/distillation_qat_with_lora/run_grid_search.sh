#!/bin/bash
set -euo pipefail

# ── Usage ───────────────────────────────────────────────────────────
# Grid search (default):
#   ./run_grid_search.sh
#
# Run specific configurations:
#   ./run_grid_search.sh --configs run_configs.txt
#
# Compare FQ_STRETCHED_LORA vs FQ_LORA with the same configs:
#   ./run_grid_search.sh --configs run_configs.txt --compression_format FQ_STRETCHED_LORA
#   ./run_grid_search.sh --configs run_configs.txt --compression_format FQ_LORA
#
# Debug mode (separate MLflow DB, won't pollute production runs):
#   ./run_grid_search.sh --debug
#   ./run_grid_search.sh --debug --configs run_configs.txt
#
# run_configs.txt format (one config per line, '#' comments allowed):
#   fq_lr  lora_lr  fq_wd  lora_wd  warmup  constant  cosine
#   1e-3   1e-3     0      1e-4     1       1         13
#   1e-5   1e-3     1e-3   1e-4     0       0         15
# ────────────────────────────────────────────────────────────────────

PRETRAINED="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="output_llama_1b"
LOG_FILE="grid_search.log"
CONFIGS_FILE=""
DEBUG_FLAG=""
COMPRESSION_FORMAT="FQ_STRETCHED_LORA"
USE_AUTOGRAD_QUANTIZE=""
SE_INIT=false

# ── Parse CLI arguments ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)   CONFIGS_FILE="$2"; shift 2 ;;
        --pretrained) PRETRAINED="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --log_file)   LOG_FILE="$2"; shift 2 ;;
        --debug)      DEBUG_FLAG="--debug"; shift ;;
        --compression_format) COMPRESSION_FORMAT="$2"; shift 2 ;;
        --use_autograd_quantize) USE_AUTOGRAD_QUANTIZE="--use_autograd_quantize"; shift ;;
        --se-init) SE_INIT=true; shift ;;
        -h|--help)
            sed -n '3,18p' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Determine MLflow DB name based on debug mode.
if [[ -n "$DEBUG_FLAG" ]]; then
    MLFLOW_DB="mlflow_debug.db"
    echo "** DEBUG MODE — results go to ${OUTPUT_DIR}/${MLFLOW_DB} **"
else
    MLFLOW_DB="mlflow.db"
fi

# Determine short format tag for run names.
case "$COMPRESSION_FORMAT" in
    FQ_STRETCHED_LORA*) FMT_TAG="seq_abs" ;;
    FQ_LORA*)           FMT_TAG="fql" ;;
    *)                  FMT_TAG="$(echo "$COMPRESSION_FORMAT" | tr '[:upper:]' '[:lower:]')" ;;
esac

# Append autograd tag to format tag when enabled.
if [[ -n "$USE_AUTOGRAD_QUANTIZE" ]]; then
    FMT_TAG="${FMT_TAG}_ag"
fi

# Append SE tag when using Scale Estimation init.
if [[ "$SE_INIT" == true ]]; then
    FMT_TAG="${FMT_TAG}_se"
    BASIC_INIT_FLAG=""
else
    BASIC_INIT_FLAG="--basic_init"
fi

# ── Helper: run a single configuration ──────────────────────────────
run_config() {
    local fq_lr="$1" lora_lr="$2" fq_wd="$3" lora_wd="$4"
    local warmup="$5" constant="$6" cosine="$7"
    local idx="$8" total="$9"

    RUN_NAME="${FMT_TAG}_fq${fq_lr}_lora${lora_lr}_fqwd${fq_wd}_lorawd${lora_wd}_w${warmup}_c${constant}_cos${cosine}"

    echo "============================================================"
    echo "[${idx}/${total}] ${RUN_NAME}"
    echo "  fq_lr=${fq_lr}  lora_lr=${lora_lr}  fq_wd=${fq_wd}  lora_wd=${lora_wd}  warmup=${warmup}  constant=${constant}  cosine=${cosine}"
    echo "  compression_format=${COMPRESSION_FORMAT}"
    echo "  output_dir=$(realpath -m "$OUTPUT_DIR")"
    echo "  log_file - $(realpath -m "$LOG_FILE")"
    echo "============================================================"

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
        --compression_format "$COMPRESSION_FORMAT" \
        $BASIC_INIT_FLAG \
        $USE_AUTOGRAD_QUANTIZE \
        $DEBUG_FLAG \
        >> "$LOG_FILE" 2>&1
}

# ═════════════════════════════════════════════════════════════════════
# Mode 1: Run explicit configurations from file
# ═════════════════════════════════════════════════════════════════════
if [[ -n "$CONFIGS_FILE" ]]; then
    if [[ ! -f "$CONFIGS_FILE" ]]; then
        echo "ERROR: configs file not found: $CONFIGS_FILE"
        exit 1
    fi

    # Read configs (skip blank lines and full-line comments, strip inline comments)
    mapfile -t lines < <(sed 's/#.*//' "$CONFIGS_FILE" | grep -vE '^\s*$')
    total=${#lines[@]}
    echo "Running ${total} explicit configuration(s) from ${CONFIGS_FILE}"

    current=0
    for line in "${lines[@]}"; do
        read -r fq_lr lora_lr fq_wd lora_wd warmup constant cosine <<< "$line"
        current=$((current + 1))
        run_config "$fq_lr" "$lora_lr" "$fq_wd" "$lora_wd" "$warmup" "$constant" "$cosine" "$current" "$total"
    done

    echo ""
    echo "Explicit configs complete -- ${current}/${total} runs."
    echo "View MLflow UI:  mlflow ui --backend-store-uri sqlite:///$(realpath -m "${OUTPUT_DIR}/${MLFLOW_DB}")"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════
# Mode 2: Full grid search (default)
# ═════════════════════════════════════════════════════════════════════

# ── Hyperparameter grid ─────────────────────────────────────────────
FQ_LRS=(0 1e-3 1e-5)
LORA_LRS=(0 1e-3 1e-5)
FQ_WEIGHT_DECAYS=(0 1e-3)
LORA_WEIGHT_DECAYS=(1e-4)

# Each row: warmup_epochs  constant_epochs  cosine_epochs
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
    if [[ "$fq_lr" == "0" && "$lora_lr" == "0" ]]; then continue; fi

    read -r warmup constant cosine <<< "$sched"
    current=$((current + 1))

    run_config "$fq_lr" "$lora_lr" "$fq_wd" "$lora_wd" "$warmup" "$constant" "$cosine" "$current" "$total_configs"

done
done
done
done
done

echo ""
echo "Grid search complete -- ${current}/${total_configs} runs."
echo "View MLflow UI:  mlflow ui --backend-store-uri sqlite:///$(realpath -m "${OUTPUT_DIR}/${MLFLOW_DB}")"
