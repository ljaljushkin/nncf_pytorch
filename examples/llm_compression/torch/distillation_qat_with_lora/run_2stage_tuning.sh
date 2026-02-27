#!/bin/bash
set -euo pipefail

# ── 2-Stage Tuning: first tune 2-bit layers, then tune 4-bit layers ──
#
# Usage:
#   ./run_2stage_tuning.sh                    # run both stages with defaults
#   ./run_2stage_tuning.sh --stage 1          # run only stage 1 (2-bit)
#   ./run_2stage_tuning.sh --stage 2          # run only stage 2 (4-bit), requires stage 1 checkpoint
#   ./run_2stage_tuning.sh --debug            # debug mode (separate MLflow DB)
#   ./run_2stage_tuning.sh --pretrained Qwen/Qwen3-8B --gradient_checkpointing
#
# Stage 1: Tune only 2-bit layers (--tune_bits 2) for STAGE1_COSINE_EPOCHS epochs.
#           Saves checkpoint to <output_dir>/last/nncf_checkpoint_epoch<N>.pth
#
# Stage 2: Load the stage-1 checkpoint and tune only 4-bit layers (--tune_bits 4)
#           for STAGE2_COSINE_EPOCHS epochs. 2-bit adapters remain frozen.
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────
PRETRAINED="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="output"
LOG_FILE="tune_2stage.log"
COMPRESSION_FORMAT="FQ_STRETCHED_LORA"
LORA_RANK=64

# Stage 1 hyperparameters (2-bit tuning)
STAGE1_FQ_LR="1e-3"
STAGE1_LORA_LR="1e-3"
STAGE1_FQ_WD="1e-3"
STAGE1_LORA_WD="1e-4"
STAGE1_WARMUP="0"
STAGE1_CONSTANT="0"
STAGE1_COSINE="1"
STAGE1_MIN_LR_RATIO="0.1"

# Stage 2 hyperparameters (4-bit tuning)
STAGE2_FQ_LR="1e-3"
STAGE2_LORA_LR="1e-3"
STAGE2_FQ_WD="1e-3"
STAGE2_LORA_WD="1e-4"
STAGE2_WARMUP="0"
STAGE2_CONSTANT="0"
STAGE2_COSINE="1"
STAGE2_MIN_LR_RATIO="0.1"

# Common training params
NUM_TRAIN_SAMPLES=512
TRAIN_SEQLEN=512
BATCH_SIZE=8

# Control which stages to run (0 = both, 1 = stage 1 only, 2 = stage 2 only)
RUN_STAGE=0

# Extra flags
DEBUG_FLAG=""
BASIC_INIT_FLAG=""
USE_AUTOGRAD_QUANTIZE=""
GRADIENT_CHECKPOINTING=""

# Explicitly provided stage-2 init checkpoint (overrides auto-detection)
STAGE2_INIT_CKPT=""

# ── Parse CLI arguments ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained)               PRETRAINED="$2"; shift 2 ;;
        --output_dir)               OUTPUT_DIR="$2"; shift 2 ;;
        --log_file)                 LOG_FILE="$2"; shift 2 ;;
        --compression_format)       COMPRESSION_FORMAT="$2"; shift 2 ;;
        --lora_rank)                LORA_RANK="$2"; shift 2 ;;
        --num_train_samples)        NUM_TRAIN_SAMPLES="$2"; shift 2 ;;
        --train_seqlen)             TRAIN_SEQLEN="$2"; shift 2 ;;
        --batch_size)               BATCH_SIZE="$2"; shift 2 ;;
        --stage)                    RUN_STAGE="$2"; shift 2 ;;
        --stage1_fq_lr)             STAGE1_FQ_LR="$2"; shift 2 ;;
        --stage1_lora_lr)           STAGE1_LORA_LR="$2"; shift 2 ;;
        --stage1_fq_wd)             STAGE1_FQ_WD="$2"; shift 2 ;;
        --stage1_lora_wd)           STAGE1_LORA_WD="$2"; shift 2 ;;
        --stage1_cosine)            STAGE1_COSINE="$2"; shift 2 ;;
        --stage1_constant)          STAGE1_CONSTANT="$2"; shift 2 ;;
        --stage1_warmup)            STAGE1_WARMUP="$2"; shift 2 ;;
        --stage1_min_lr_ratio)      STAGE1_MIN_LR_RATIO="$2"; shift 2 ;;
        --stage2_fq_lr)             STAGE2_FQ_LR="$2"; shift 2 ;;
        --stage2_lora_lr)           STAGE2_LORA_LR="$2"; shift 2 ;;
        --stage2_fq_wd)             STAGE2_FQ_WD="$2"; shift 2 ;;
        --stage2_lora_wd)           STAGE2_LORA_WD="$2"; shift 2 ;;
        --stage2_cosine)            STAGE2_COSINE="$2"; shift 2 ;;
        --stage2_constant)          STAGE2_CONSTANT="$2"; shift 2 ;;
        --stage2_warmup)            STAGE2_WARMUP="$2"; shift 2 ;;
        --stage2_min_lr_ratio)      STAGE2_MIN_LR_RATIO="$2"; shift 2 ;;
        --stage2_init_ckpt)         STAGE2_INIT_CKPT="$2"; shift 2 ;;
        --debug)                    DEBUG_FLAG="--debug"; shift ;;
        --basic_init)               BASIC_INIT_FLAG="--basic_init"; shift ;;
        --use_autograd_quantize)    USE_AUTOGRAD_QUANTIZE="--use_autograd_quantize"; shift ;;
        --gradient_checkpointing)   GRADIENT_CHECKPOINTING="--gradient_checkpointing"; shift ;;
        -h|--help)
            sed -n '3,17p' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LAST_DIR="${OUTPUT_DIR}/last"
STAGE1_FAILED=false
STAGE2_FAILED=false

# ── Stage 1: Tune 2-bit layers ──────────────────────────────────────
if [[ "$RUN_STAGE" == "0" || "$RUN_STAGE" == "1" ]]; then
    STAGE1_EPOCHS=$((STAGE1_CONSTANT + STAGE1_COSINE))
    STAGE1_RUN_NAME="2stage_s1_2bit_fq${STAGE1_FQ_LR}_lora${STAGE1_LORA_LR}_cos${STAGE1_COSINE}"

    echo "═══════════════════════════════════════════════════════════════"
    echo "  STAGE 1: Tuning 2-bit layers (${STAGE1_EPOCHS} epochs)"
    echo "  fq_lr=${STAGE1_FQ_LR}  lora_lr=${STAGE1_LORA_LR}"
    echo "  output_dir=$(realpath -m "$OUTPUT_DIR")"
    echo "═══════════════════════════════════════════════════════════════"

    if python "$SCRIPT_DIR/main.py" \
        --pretrained "$PRETRAINED" \
        --lora_rank "$LORA_RANK" \
        --num_train_samples "$NUM_TRAIN_SAMPLES" \
        --train_seqlen "$TRAIN_SEQLEN" \
        --batch_size "$BATCH_SIZE" \
        --fq_lr "$STAGE1_FQ_LR" \
        --lora_lr "$STAGE1_LORA_LR" \
        --fq_weight_decay "$STAGE1_FQ_WD" \
        --lora_weight_decay "$STAGE1_LORA_WD" \
        --warmup_epochs "$STAGE1_WARMUP" \
        --constant_epochs "$STAGE1_CONSTANT" \
        --cosine_epochs "$STAGE1_COSINE" \
        --min_lr_ratio "$STAGE1_MIN_LR_RATIO" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$STAGE1_RUN_NAME" \
        --compression_format "$COMPRESSION_FORMAT" \
        --tune_bits 2 \
        --save_epochs "$STAGE1_EPOCHS" \
        $BASIC_INIT_FLAG \
        $USE_AUTOGRAD_QUANTIZE \
        $GRADIENT_CHECKPOINTING \
        $DEBUG_FLAG \
        >> "$LOG_FILE" 2>&1; then
        echo "Stage 1 complete. Checkpoint: ${LAST_DIR}/nncf_checkpoint_epoch${STAGE1_EPOCHS}.pth"
    else
        echo "Stage 1 FAILED (exit code $?) — see ${LOG_FILE} for details"
        STAGE1_FAILED=true
    fi
fi

# ── Stage 2: Tune 4-bit layers (resume from stage 1 checkpoint) ─────
if [[ "$RUN_STAGE" == "0" || "$RUN_STAGE" == "2" ]]; then
    # Determine the stage-1 checkpoint to use as init for stage 2.
    if [[ -n "$STAGE2_INIT_CKPT" ]]; then
        S2_INIT_CKPT="$STAGE2_INIT_CKPT"
    else
        # Auto-detect: use the last checkpoint from stage 1.
        STAGE1_EPOCHS=$((STAGE1_CONSTANT + STAGE1_COSINE))
        S2_INIT_CKPT="${LAST_DIR}/nncf_checkpoint_epoch${STAGE1_EPOCHS}.pth"
    fi

    if [[ ! -f "$S2_INIT_CKPT" ]]; then
        echo "ERROR: Stage-2 init checkpoint not found: $S2_INIT_CKPT"
        echo "Run stage 1 first, or provide --stage2_init_ckpt <path>."
        exit 1
    fi

    STAGE2_EPOCHS=$((STAGE2_CONSTANT + STAGE2_COSINE))
    STAGE2_RUN_NAME="2stage_s2_4bit_fq${STAGE2_FQ_LR}_lora${STAGE2_LORA_LR}_cos${STAGE2_COSINE}"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  STAGE 2: Tuning 4-bit layers (${STAGE2_EPOCHS} epochs)"
    echo "  fq_lr=${STAGE2_FQ_LR}  lora_lr=${STAGE2_LORA_LR}"
    echo "  init_ckpt=${S2_INIT_CKPT}"
    echo "  output_dir=$(realpath -m "$OUTPUT_DIR")"
    echo "═══════════════════════════════════════════════════════════════"

    if python "$SCRIPT_DIR/main.py" \
        --pretrained "$PRETRAINED" \
        --lora_rank "$LORA_RANK" \
        --num_train_samples "$NUM_TRAIN_SAMPLES" \
        --train_seqlen "$TRAIN_SEQLEN" \
        --batch_size "$BATCH_SIZE" \
        --fq_lr "$STAGE2_FQ_LR" \
        --lora_lr "$STAGE2_LORA_LR" \
        --fq_weight_decay "$STAGE2_FQ_WD" \
        --lora_weight_decay "$STAGE2_LORA_WD" \
        --warmup_epochs "$STAGE2_WARMUP" \
        --constant_epochs "$STAGE2_CONSTANT" \
        --cosine_epochs "$STAGE2_COSINE" \
        --min_lr_ratio "$STAGE2_MIN_LR_RATIO" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$STAGE2_RUN_NAME" \
        --compression_format "$COMPRESSION_FORMAT" \
        --tune_bits 4 \
        --init_ckpt "$S2_INIT_CKPT" \
        --save_epochs "$STAGE2_EPOCHS" \
        $BASIC_INIT_FLAG \
        $USE_AUTOGRAD_QUANTIZE \
        $GRADIENT_CHECKPOINTING \
        $DEBUG_FLAG \
        >> "$LOG_FILE" 2>&1; then
        echo "Stage 2 complete. Checkpoint: ${LAST_DIR}/nncf_checkpoint_epoch${STAGE2_EPOCHS}.pth"
    else
        echo "Stage 2 FAILED (exit code $?) — see ${LOG_FILE} for details"
        STAGE2_FAILED=true
    fi
fi

echo ""
if [[ "$STAGE1_FAILED" == true || "$STAGE2_FAILED" == true ]]; then
    echo "2-stage tuning finished WITH ERRORS:"
    [[ "$STAGE1_FAILED" == true ]] && echo "  - Stage 1 (2-bit) FAILED"
    [[ "$STAGE2_FAILED" == true ]] && echo "  - Stage 2 (4-bit) FAILED"
else
    echo "2-stage tuning finished successfully."
fi
echo "Log: $(realpath -m "$LOG_FILE")"
