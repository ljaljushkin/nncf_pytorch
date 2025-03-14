#!/bin/bash
set -e

printf '##################################\n'
printf '########  Installing environment\n'
printf '##################################\n'

ENV_NAME="env_torch_ref"

mkdir -p $HOME/MODEL_DIR

# rm -rf $ENV_NAME
# python3.11 -m venv $ENV_NAME
. $ENV_NAME/bin/activate
# pip install -U pip
# pip install -r requirements.txt
# pip install -e ../../../

printf '##################################\n'
printf '########  Create NNCF checkpoint with 4bit FQ+LoRA \n'
printf '##################################\n'

BASE_MODEL="HuggingFaceTB/SmolLM-1.7B-Instruct"
MODEL_NAME="SmolLM-1_7B-Instruct"
MAX_LENGTH=2048

INIT_DIR="FQ_emb_head_int8_int4_asym_rank256_gs64_test"
EXP_NAME="SmolL_lr5e-04_fqlr5e-05_wd5e-04"
python add_FQ_with_LoRA.py -m $BASE_MODEL -s $INIT_DIR


printf '##################################\n'
printf '########  Quantization-aware tuning of lora adapters and quantization scales \n'
printf '##################################\n'

# Nsight Compute Profiler
# sudo -HE PYTHONPATH=/home/nlyaly/projects/nncf2 /usr/local/cuda-12.4/bin/ncu -k q_scale_per_weight_channel_cuda_backward_kernel --set full -f -o bw_kernel_tinb_1024 /home/nlyaly/env/nncf2-py/bin/python tools/benchmark_quantize_layers.py

# Nsight System Profiler
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true \
# -o kernel_1024_threads_in_block --force-overwrite true \
tune_command_template="PYTHONIOENCODING=utf-8 \
python tune_fq_lora_chat.py \
--nncf_ckpt_dir=$HOME/MODEL_DIR/$MODEL_NAME/$INIT_DIR \
--base_model=$BASE_MODEL \
--model_seqlen=\$model_seqlen \
--adam_beta1=0.90 \
--adam_beta2=0.999 \
--batch_size=\$batch_size \
--microbatch_size=\$microbatch_size \
--trust_remote_code  \
--nsamples=\$nsamples \
--weight_decay=\$weight_decay \
--dataset=\$dataset \
--lr=\$lr \
--fq_lr=\$fq_lr \
--epochs=\$epochs \
--finetune_dtype=bfloat16 \
--device_map=auto \
--eval_model_seqlen=$MAX_LENGTH \
--exp_name $EXP_NAME \
--mlflow"

weight_decays=5e-4 #2e-4 1e-2) #(0 1e-5 1e-2)
model_seqlen=1024
batch_sizes=32 #(128 64) #32
microbatch_size=2 #2 #2
list_nsamples=1024 #128
dataset=wikitext2
lrs=5e-4
fq_lrs=5e-5
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
                        echo "Running: $command"
                        mkdir -p logs
                        eval $command 2>&1 | tee -a "logs/tune_${MODEL_NAME}_$(date '+%Y-%m-%d_%H:%M:%S').log" # _$(date '+%Y-%m-%d_%H:%M:%S')
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
# PYTHONIOENCODING=utf-8 ./eval_slm.sh $BASE_MODEL $MODEL_NAME $INIT_DIR $EXP_NAME $MAX_LENGTH