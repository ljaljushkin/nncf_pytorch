#!/bin/bash

set -e


printf '##################################\n'
printf '########  Installing environment\n'
printf '##################################\n'

mkdir -p $HOME/MODEL_DIR
# rm -rf env
# python3.11 -m venv env
# . env/bin/activate
# pip install -U pip
# pip install -r requirements.txt
# pip install ../../../


printf '##################################\n'
printf '########  Create NNCF checkpoint with 4bit FQ+LoRA \n'
printf '##################################\n'

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

# INIT_DIR="FQ_emb_head_int8_sym_int4_sym_rank256_gs-1_demo"
# INIT_DIR="FQ_emb_head_int8_sym_int4_sym_rank256_gs512_demo"

BASE_MODEL="microsoft/Phi-3.5-mini-instruct"
MODEL_NAME="Phi-3_5-mini-instruct"
MAX_LENGTH=4096
INIT_DIR="FQ_emb_head_int8_asym_int4_asym_rank256_gs64_demo"
EXP_NAME="Phi-3_lr5e-04_fqlr5e-05_wd5e-04_tune_all"
PYTHONIOENCODING=utf-8 python add_FQ_with_LoRA.py -m $BASE_MODEL -s $INIT_DIR


printf '##################################\n'
printf '########  Quantization-aware tuning of lora adapters and quantization scales \n'
printf '##################################\n'

tune_command_template="PYTHONIOENCODING=utf-8 python tune_fq_lora.py \
--nncf_ckpt_dir=$HOME/MODEL_DIR/$MODEL_NAME/$INIT_DIR \
--base_model=$BASE_MODEL \
--model_seqlen=\$model_seqlen \
--adam_beta1=0.90  \
--adam_beta2=0.999  \
--batch_size=\$batch_size \
--microbatch_size=\$microbatch_size \
--trust_remote_code  \
--nsamples=\$nsamples \
--weight_decay=\$weight_decay \
--dataset=\$dataset \
--lr=\$lr \
--fq_lr=\${fq_lr} \
--epochs=\$epochs \
--finetune_dtype=bfloat16 \
--device_map=auto \
--eval_model_seqlen=$MAX_LENGTH \
--exp_name $EXP_NAME \
--mlflow"

weight_decays=1e-4 #2e-4 1e-2) #(0 1e-5 1e-2)
model_seqlen=1024
batch_sizes=32 #(128 64) #32
microbatch_size=2 #2 #2
list_nsamples=1024 #128
dataset=wikitext2
lrs=1e-4
fq_lrs=1e-5
list_epochs=32 #2 #(8 16 32)

for batch_size in "${batch_sizes[@]}"
do
    for lr in "${lrs[@]}"
    do
        for weight_decay in "${weight_decays[@]}"
        do
            for fq_lr in "${fq_lrs[@]}"
            do
                for nsamples in "${list_nsamples[@]}"
                do
                    for epochs in "${list_epochs[@]}"
                    do
                        export model_seqlen batch_size microbatch_size nsamples weight_decay dataset lr fq_lr epochs
                        command=$(echo $tune_command_template | envsubst)
                        mkdir -p logs
                        echo "Running: $command"
                        eval $command 2>&1 | tee -a "logs/tune_${MODEL_NAME}_$(date '+%Y-%m-%d_%H:%M:%S').log"
                    done
                done
            done
        done
    done
done


printf '##################################\n'
printf '########  Evaluation of the best checkpoint'
printf '##################################\n'

unset CUDA_VISIBLE_DEVICES
PYTHONIOENCODING=utf-8 ./eval.sh $BASE_MODEL $MODEL_NAME $INIT_DIR $EXP_NAME $MAX_LENGTH