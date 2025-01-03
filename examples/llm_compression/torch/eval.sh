MODEL_DIR="$HOME/MODEL_DIR"

# BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME="Phi-3-mini-4k-instruct"

# BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
# MODEL_NAME="Qwen2_5-3B-Instruct"

# BASE_MODEL="google/gemma-2-2b-it"
# MODEL_NAME="gemma-2-2b-it"

# BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME="Meta-Llama-3-8B-Instruct"

# BASE_MODEL="mistralai/Mistral-7B-v0.3"
# MODEL_NAME="Mistral-7B-v0_3"

# BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="Llama-3_2-1B-Instruct"

# BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME="Llama-3_2-3B-Instruct"

# BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME="Meta-Llama-3-8B-Instruct"

BASE_MODEL=${1:-"microsoft/Phi-3.5-mini-instruct"}
MODEL_NAME=${2:-"Phi-3_5-mini-instruct"}
INIT_DIR=${3:-"FQ_emb_head_int8_asym_int4_asym_rank256_gs64_demo"}
EXP_DIR=${4:-"Phi-3_lr5e-04_fqlr5e-05_wd5e-04_tune_all"}
MAX_LENGTH=${5:-4096}

echo $@
NNCF_CKPT_DIR=$MODEL_DIR/$MODEL_NAME/$INIT_DIR/$EXP_DIR
printf "NNCF_CKPT_DIR=$NNCF_CKPT_DIR\n"


TASK="wikitext"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="gsm8k"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot=8 > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="ifeval"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="arc_challenge"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="hellaswag"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="mmlu"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot 5 --batch_size 4 > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# # pip install whowhatbench@git+https://github.com/andreyanufr/openvino.genai.git@837294cb21a9bb408faa346ddde287ea748ee22c#subdirectory=tools/who_what_benchmark
cd ../nncf
TASK="WWB"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 python wwb_eval.py -m=$BASE_MODEL -n=$NNCF_CKPT_DIR > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"
cd -
