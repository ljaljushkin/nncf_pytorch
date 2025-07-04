# CUDA_VISIBLE_DEVICES=0 python eval_ref.py -m HuggingFaceTB/SmolLM-1.7B-Instruct -f out_HuggingFaceTB_SmolLM-1_7B-Instruct_alpha_gs64_asym -s 2048 > /dev/null 2>&1 &
# echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=4 python eval_ref.py -m microsoft/Phi-3-mini-4k-instruct -f out_microsoft_Phi-3-mini-4k-instruct_alpha_gs64_asym > /dev/null 2>&1 &
# echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=5 python eval_ref.py -m meta-llama/Llama-3.2-1B-Instruct -f out_meta-llama_Llama-3_2-1B-Instruct_alpha_gs64_asym > /dev/null 2>&1 &
# echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=0 python eval_ref.py -m google/gemma-2-2b-it -f out_google_gemma-2-2b-it_alpha_gs64_asym > /dev/null 2>&1 &
# echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=5 python eval_ref.py -m meta-llama/Llama-3.2-3B-Instruct -f out_meta-llama_Llama-3_2-3B-Instruct_alpha_gs64_asym > /dev/null 2>&1 &
# echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=6 python eval_ref.py -m microsoft/Phi-3.5-mini-instruct -f out_microsoft_Phi-3_5-mini-instruct_alpha_gs64_asym > /dev/null 2>&1 &
# echo "Started eval with PID $!"
CUDA_VISIBLE_DEVICES=0 python eval_ref.py -m meta-llama/Meta-Llama-3-8B-Instruct -f out_meta-llama_Meta-Llama-3-8B-Instruct_repro_r256 > /dev/null 2>&1 &
echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=1 python eval_ref.py -m mistralai/Mistral-7B-v0.3 -f out_mistralai_Mistral-7B-v0_3_alpha_gs64_asym  > /dev/null 2>&1 &
# echo "Started eval with PID $!"
# CUDA_VISIBLE_DEVICES=5 python eval_ref.py -m Qwen/Qwen2.5-3B-Instruct -f out_Qwen_Qwen2_5-3B-Instruct_alpha_gs64_asym > /dev/null 2>&1 &
# echo "Started eval with PID $!"






