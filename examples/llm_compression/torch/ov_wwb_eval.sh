ROOT_DIR=/local_ssd2/nlyalyus/MODEL_DIR/DeepSeek-R1-Distill-Qwen-1_5B/
TARGET_DIR=$ROOT_DIR/FQ_emb_head_int8_asym_int4_asym_rank256_gs32_se/DS_Qwen_lr5e-04_fqlr5e-05_wd5e-04/best_wwb_ckpt/exported

wwb \
--gt-data $ROOT_DIR/ref_qa_chat.csv \
--target-model $TARGET_DIR \
--model-type text \
--output $TARGET_DIR/wwb_no_kv_cache_chat \
--ov-config $ROOT_DIR/ov_config.json \
--language cn \
--chat-template
