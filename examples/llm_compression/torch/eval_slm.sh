MODEL_DIR="$HOME/MODEL_DIR"

BASE_MODEL=${1:-"HuggingFaceTB/SmolLM-1.7B-Instruct"}
MODEL_NAME=${2:-"SmolLM-1_7B-Instruct"}
INIT_DIR=${3:-"FQ_emb_head_int8_asym_int4_asym_rank256_gs64_demo"}
EXP_DIR=${4:-"SmolL_lr5e-04_fqlr5e-05_wd5e-04_tune_all"}
MAX_LENGTH=${5:-2048}

echo $@
NNCF_CKPT_DIR=$MODEL_DIR/$MODEL_NAME/$INIT_DIR/$EXP_DIR
printf "NNCF_CKPT_DIR=$NNCF_CKPT_DIR\n"

TASK="wikitext"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

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
# CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$BASE_MODEL,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot 5 --batch_size 4 > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# pip install whowhatbench@git+https://github.com/andreyanufr/openvino.genai.git@837294cb21a9bb408faa346ddde287ea748ee22c#subdirectory=tools/who_what_benchmark
TASK="WWB"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 python wwb_eval.py -m=$BASE_MODEL -n=$NNCF_CKPT_DIR > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"
