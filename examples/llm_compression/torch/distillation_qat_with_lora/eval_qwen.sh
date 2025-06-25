
# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --tasks gpqa --apply_chat_template --batch_size 8 --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4
# CUDA_VISIBLE_DEVICES=2 lm_eval --model hf --tasks gpqa --apply_chat_template --batch_size 16 --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct
# CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct


# CUDA_VISIBLE_DEVICES=4 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora/out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_chat_alpaca_755_manual_chat

# cd /local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora/out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_gsm8k
# CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=. --output_path hellaswag.json


# cd /local_ssd2/nlyalyus/projects/nncf/tmp_autoround/Qwen2.5-1.5B-Instruct-w4g128
# CUDA_VISIBLE_DEVICES=7 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=. --output_path hellaswag.json
# CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --tasks gpqa --batch_size 8 --model_args pretrained=. --output_path gpqa.json --apply_chat_template

# CUDA_VISIBLE_DEVICES=0 auto_round --model microsoft/Phi-3-mini-4k-instruct --bits 4 --group_size 128 --format auto_gptq --output_dir auto_round_phi3_4k
# CUDA_VISIBLE_DEVICES=1 auto_round --model meta-llama/Llama-3.2-3B-Instruct --bits 4 --group_size 128 --format auto_gptq --output_dir auto_round_llama_3_2_3b

# cd auto_round_phi3_4k/Phi-3-mini-4k-instruct-w4g128
# task=hellaswag
# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --tasks $task --batch_size 16 --model_args pretrained=. --output_path $task.json > $(pwd)/$task.log 2>&1 &
# echo "Started eval with PID=$! log $(pwd)/$task.log"
# task=wikitext
# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --tasks $task --model_args pretrained=. --output_path $task.json > $(pwd)/$task.log 2>&1 &
# echo "Started eval with PID=$! log $(pwd)/$task.log"
# task=gsm8k
# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --tasks $task --model_args pretrained=. --apply_chat_template --output_path $task.json > $(pwd)/$task.log 2>&1 &
# echo "Started eval with PID=$! log $(pwd)/$task.log"
# cd -

# cd auto_round_llama_3_2_3b/Llama-3.2-3B-Instruct-w4g128
# task=hellaswag
# CUDA_VISIBLE_DEVICES=2 lm_eval --model hf --tasks $task --batch_size 16 --model_args pretrained=. --output_path $task.json > $(pwd)/$task.log 2>&1 &
# echo "Started eval with PID=$! log $(pwd)/$task.log"
# task=wikitext
# CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --tasks $task --model_args pretrained=. --output_path $task.json > $(pwd)/$task.log 2>&1 &
# echo "Started eval with PID=$! log $(pwd)/$task.log"
# task=gsm8k
# CUDA_VISIBLE_DEVICES=4 lm_eval --model hf --tasks $task --model_args pretrained=. --apply_chat_template --output_path $task.json > $(pwd)/$task.log 2>&1 &
# echo "Started eval with PID=$! log $(pwd)/$task.log"
# cd -

# CUDA_VISIBLE_DEVICES=0 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f Qwen_Qwen2_5-1_5B-Instruct_repro_r128_alpha # > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=1 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f Qwen_Qwen2_5-1_5B-Instruct_repro_r128_alpha_t2 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"



# CUDA_VISIBLE_DEVICES=2 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f meta-llama_Llama-3_2-3B-Instruct_repro_r256_t2 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=3 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f meta-llama_Llama-3_2-3B-Instruct_repro_r256_alpha > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=4 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f meta-llama_Llama-3_2-3B-Instruct_repro_r32 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=5 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f meta-llama_Llama-3_2-3B-Instruct_repro_r32_alpha_t2 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=6 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f meta-llama_Llama-3_2-3B-Instruct_repro_r32_t2 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=7 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f meta-llama_Llama-3_2-3B-Instruct_repro_r32_alpha > /dev/null 2>&1 &
# echo "Started eval with PID=$!"



# CUDA_VISIBLE_DEVICES=0 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r256_log > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=1 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r256_alpha_log > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=2 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r128 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=3 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r128_alpha > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=4 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r32 > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=5 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r32_alpha > /dev/null 2>&1 &
# echo "Started eval with PID=$!"


# model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "microsoft/Phi-3-mini-4k-instruct"

# meta-llama_Llama-3_2-3B-Instruct_repro_r128
# meta-llama_Llama-3_2-3B-Instruct_repro_r128_t2
# Qwen_Qwen2_5-1_5B-Instruct_repro_r128
# Qwen_Qwen2_5-1_5B-Instruct_repro_r128_t2
# microsoft_Phi-3-mini-4k-instruct_repro_r32_t2
# microsoft_Phi-3-mini-4k-instruct_repro_r32


CUDA_VISIBLE_DEVICES=6 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f out_meta-llama_Llama-3_2-3B-Instruct_repro_r256_log_int8_head > /dev/null 2>&1 &
echo "Started eval with PID=$!"
CUDA_VISIBLE_DEVICES=7 python eval.py -m meta-llama/Llama-3.2-3B-Instruct -f out_meta-llama_Llama-3_2-3B-Instruct_repro_r256_alpha_log_int8_head > /dev/null 2>&1 &
echo "Started eval with PID=$!"

CUDA_VISIBLE_DEVICES=4 python eval.py -m microsoft/Phi-3.5-mini-instruct -f out_microsoft_Phi-3_5-mini-instruct_repro_r256_log_int8_head > /dev/null 2>&1 &
echo "Started eval with PID=$!"
CUDA_VISIBLE_DEVICES=5 python eval.py -m microsoft/Phi-3.5-mini-instruct -f out_microsoft_Phi-3_5-mini-instruct_repro_r256_alpha_log_int8_head > /dev/null 2>&1 &
echo "Started eval with PID=$!"

# CUDA_VISIBLE_DEVICES=6 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r256_alpha_log_int8_head > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
# CUDA_VISIBLE_DEVICES=7 python eval.py -m Qwen/Qwen2.5-1.5B-Instruct -f out_Qwen_Qwen2_5-1_5B-Instruct_repro_r256_log_int8_head > /dev/null 2>&1 &
# echo "Started eval with PID=$!"
