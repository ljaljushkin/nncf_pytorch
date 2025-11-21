import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Mxfp4Config

import nncf

DIR_NAME = "int4_sym_gs32_experts"
compression_args = dict(
    mode=nncf.CompressWeightsMode.INT4_SYM,
    group_size=32,
    ignored_scope=nncf.IgnoredScope(patterns=[r".*self_attn.*", r".*router.*"]),
    backup_mode=nncf.BackupMode.NONE,
    scale_estimation=True,
)

model_id = "tiny-random/gpt-oss-mxfp4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# mxfp4_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=Mxfp4Config(dequantize=True)
)

inputs = {k: v.to(model.device) for k, v in model.dummy_inputs.items()}
dataset = nncf.Dataset([inputs])
compressed_model = nncf.compress_weights(model, dataset=dataset, **compression_args)
