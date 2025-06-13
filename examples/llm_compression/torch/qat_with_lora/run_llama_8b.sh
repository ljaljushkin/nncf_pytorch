# CUDA_VISIBLE_DEVICES=0,1 python main.py --pretrained meta-llama/Meta-Llama-3-8B-Instruct --output_dir out_llama_8B_wiki_nt --lr 5e-5 --eval_seqlen=4096 --microbatch_size=2 > log_llama3_8B_wiki_nt.txt 2>&1
# git checkout -
# CUDA_VISIBLE_DEVICES=0,1 python main.py --pretrained meta-llama/Meta-Llama-3-8B-Instruct --output_dir out_llama_8B_wiki_ot --lr 5e-5 --eval_seqlen=4096 --microbatch_size=2 > log_llama3_8B_wiki_ot.txt 2>&1

lm_eval --model hf --model_args pretrained=tmp_autoround/Qwen2.5-1.5B-Instruct-w4g128 --tasks=gsm8k --apply_chat_template --batch_size 16