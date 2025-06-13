
# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --tasks gpqa --apply_chat_template --batch_size 8 --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4
# CUDA_VISIBLE_DEVICES=2 lm_eval --model hf --tasks gpqa --apply_chat_template --batch_size 16 --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct
# CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct


CUDA_VISIBLE_DEVICES=4 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora/out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_chat_alpaca_755_manual_chat

cd /local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora/out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_gsm8k
CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=. --output_path hellaswag.json


cd /local_ssd2/nlyalyus/projects/nncf/tmp_autoround/Qwen2.5-1.5B-Instruct-w4g128
CUDA_VISIBLE_DEVICES=7 lm_eval --model hf --tasks hellaswag --batch_size 16 --model_args pretrained=. --output_path hellaswag.json
CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --tasks gpqa --batch_size 8 --model_args pretrained=. --output_path gpqa.json --apply_chat_template

