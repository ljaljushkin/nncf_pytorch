#!/bin/bash

run_model_async() {
  local cuda=$1
  local pretrained=$2
  local lr=$3
  local extra_args=$4

  # If no pretrained model, use a custom tag
  local model_tag=$(echo $pretrained | sed 's|/|_|g' | sed 's|\.-|_|g' | sed 's|\.|_|g')

  local output_dir="out_${model_tag}_repro_chat"
  local log_file="log_${model_tag}_repro_chat.txt"

  echo "Running $pretrained on CUDA $cuda..."
  CUDA_VISIBLE_DEVICES=$cuda python main_chat.py --fast_eval --pretrained $pretrained --output_dir $output_dir --lr $lr $extra_args > $log_file 2>&1 &

  pid=$!
  echo "Started $model_tag with PID: $pid with log file: $log_file"
  pids+=($pid)
}

# Array to store PIDs
pids=()

# --pretrained microsoft/Phi-3.5-mini-instruct  --output_dir out_phi_3.5_mini_instruct_128_sym_lora --lr 1e-4 --eval_seqlen=4096 --microbatch_size 2 --batch_size 64 --lora_rank 256 --epochs 2
# --pretrained Qwen/Qwen2.5-1.5B-Instruct   --output_dir out_qwen_2.5_1.5b_ --lr 1e-4 --eval_seqlen=4096 --microbatch_size 2 --batch_size 64 --lora_rank 128 --epochs 2

# Launch models asynchronously
# REPRO
run_model_async 2 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --epochs 10 --lora_rank 128"
run_model_async 2 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --epochs 10 --lora_rank 128"
run_model_async 2 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --epochs 10 --lora_rank 128"
run_model_async 2 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --epochs 10 --lora_rank 128"
# CHAT dolly 128
# run_model_async 2 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --epochs 10 --lora_rank 128 --calib_seqlen=128 --num_train_samples=128"
# CHAT dolly 512
# run_model_async 1 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--use_dolly --eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --lora_rank 128 --calib_seqlen=512 --num_train_samples=512 --epochs=10 --resume"
# NO CHAT dolly 128
# run_model_async 2 Qwen/Qwen2.5-1.5B-Instruct 1e-4 "--use_dolly --no_chat --eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --lora_rank 128 --calib_seqlen=128 --num_train_samples=128"


# run_model_async 5 Qwen/Qwen2.5-1.5B-Instruct 5e-5 "--eval_seqlen=4096 --microbatch_size=2 --batch_size 64 --epochs 32"

# run_model_async 3 meta-llama/Llama-3.2-3B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=4"

# run_model_async 0 meta-llama/Llama-3.2-1B-Instruct 5e-5 "--eval_seqlen=4096 --microbatch_size=4"
# run_model_async 1 meta-llama/Llama-3.2-3B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2"
# run_model_async 2 HuggingFaceTB/SmolLM-1.7B-Instruct 5e-4 "--eval_seqlen=2048 --microbatch_size=4"
# run_model_async 3 microsoft/Phi-3.5-mini-instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2"
# run_model_async 4 google/gemma-2-2b-it 1e-4 "--eval_seqlen=4096 --microbatch_size=4"
# run_model_async 5 microsoft/Phi-3-mini-4k-instruct 5e-5 "--eval_seqlen=4096 --microbatch_size=4"
# run_model_async 6,7 meta-llama/Meta-Llama-3-8B-Instruct 5e-5 "--eval_seqlen=4096 --microbatch_size=2"

# run_model_async 1 Qwen/Qwen2.5-3B-Instruct 1e-4 "--eval_seqlen=4096 --microbatch_size=2"
# run_model_async 0,2 mistralai/Mistral-7B-v0.3 5e-5 "--eval_seqlen=4096 --microbatch_size=2"

echo ""
echo "All processes started. Use the following PIDs to monitor or kill if needed:"
for pid in "${pids[@]}"; do
  echo "  PID: $pid"
done