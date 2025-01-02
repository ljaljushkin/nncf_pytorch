MODEL_DIR="$HOME/MODEL_DIR"
MODEL_ID="HuggingFaceTB/SmolLM-1.7B-Instruct"
MODEL_NAME="SmolLM-1_7B-Instruct"
INIT_DIR="FQ_emb_head_int8_sym_int4_sym_rank256_gs-1"
# EXP_DIR=""

MAX_LENGTH=2048
NNCF_CKPT_DIR=$MODEL_DIR/$MODEL_NAME/$INIT_DIR/$EXP_DIR

TASK="wikitext"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

TASK="wikitext"
NNCF_CKPT_DIR=$MODEL_DIR/$MODEL_NAME
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="gsm8k"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot=8 > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="ifeval"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="arc_challenge"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="hellaswag"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="mmlu"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot 5 --batch_size 4 > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# # pip install whowhatbench@git+https://github.com/andreyanufr/openvino.genai.git@837294cb21a9bb408faa346ddde287ea748ee22c#subdirectory=tools/who_what_benchmark
# cd ../nncf
# TASK="WWB"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python wwb_eval.py -m=$MODEL_ID -n=$NNCF_CKPT_DIR > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"
# cd -
