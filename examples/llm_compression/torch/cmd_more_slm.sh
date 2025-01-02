#!/bin/bash

BASE_MODEL="HuggingFaceTB/SmolLM-1.7B-Instruct"
MODEL_NAME="SmolLM-1_7B-Instruct"

# BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME="Phi-3-mini-4k-instruct"

# BASE_MODEL="microsoft/Phi-3.5-mini-instruct"
# MODEL_NAME="Phi-3_5-mini-instruct"

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

# --nncf_ckpt_dir=$HOME/MODEL_DIR/$MODEL_NAME/FQ_emb_head_int8_asym_int4_asym_rank\${rank}_gs64 \

tune_command_template="PYTHONIOENCODING=utf-8 python tune_fq_lora.py \
--nncf_ckpt_dir=$HOME/MODEL_DIR/$MODEL_NAME/FQ_emb_head_int8_sym_int4_sym_rank256_gs512_demo \
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
--eval_model_seqlen=2048 \
--mlflow"

# --print_every_steps 1"
# --use_fast_tokenizer"


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
                        eval $command 2>&1 | tee -a "logs/tune_${MODEL_NAME}_$(date '+%Y-%m-%d_%H:%M:%S').log" # _$(date '+%Y-%m-%d_%H:%M:%S')

                        # run_commands "FQ_4bit_no_embed_svd_rank8_g64" "${ckpt_dir[@]}"
                    done
                done
            done
        done
    done
done
