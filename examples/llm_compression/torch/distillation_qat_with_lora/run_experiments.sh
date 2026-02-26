#!/bin/bash

# Output directories to process
OUTPUT_DIRS=("output_llama_1b")
PRETRAINED="meta-llama/Llama-3.2-1B-Instruct"

# Checkpoint files to evaluate
CKPT_FILES=("nncf_checkpoint_after_5th_epoch.pth" "nncf_checkpoint_after_first_epoch.pth")
# CKPT_FILES=("nncf_checkpoint_after_first_epoch.pth") #"nncf_checkpoint_svd_lora_se_2bit.pth") #"nncf_checkpoint_svd_lora_se_tune_scales.pth")

LOG_FILE="tune.log"

# --limit 2 \
# --log_samples \
# Function to run lm_eval
run_lm_eval_gsm8k() {
    local model_dir="$1"
    local log_file="$2"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"enable_thinking\":false,\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks gsm8k \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --output_path eval_results/tmp \
        --gen_kwargs do_sample=True,temperature=0.7,top_p=0.8,top_k=20,min_p=0 \
        --batch_size auto >> "$log_file" 2>&1
}

run_lm_eval() {
    local model_dir="$1"
    local log_file="$2"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks lambada_openai \
        --output_path eval_results/llama_1b_lambada \
        --batch_size auto >> "$log_file" 2>&1
}
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    CKPT_DIR="$OUTPUT_DIR/last/stripped"

    Run training
    echo "Running training with output_dir: $(realpath "$OUTPUT_DIR") and log_file: $(realpath "$LOG_FILE")"
    python main.py --pretrained $PRETRAINED --fq_lr 1e-3 --weight_decay_fq 1e-3 --lora_lr 0 --cosine_epochs 0 --constant_epochs 15 --warmup_epochs 0 --min_lr_ratio 0.1 --resume --output_dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

    # First evaluation (after training, before stripping any checkpoint)
    # echo "Running lm-eval after training..."
    # run_lm_eval "$CKPT_DIR" "$LOG_FILE"

    # Loop through checkpoint files
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        echo "Saving stripped checkpoint: $CKPT_FILE to $CKPT_DIR"
        python save_stripped.py -p $PRETRAINED -c "$OUTPUT_DIR/last/$CKPT_FILE" -o "$CKPT_DIR"

        echo "Running lm-eval for $CKPT_FILE..."
        run_lm_eval "$CKPT_DIR" "$LOG_FILE"
    done
done

echo "All experiments completed!"
