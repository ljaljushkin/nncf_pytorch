# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import sys
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from optimum.exporters.openvino.convert import export_from_model
from whowhatbench import TextEvaluator
from nncf.torch import load_from_config
from nncf.torch.model_graph_manager import get_module_by_name
from optimum.intel.openvino import OVModelForCausalLM
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.logger import set_log_file
from tune_fq_lora_chat import get_loaders
from enum import Enum, auto

class StripMode(Enum):
    NONE = auto()
    TO_FLOAT = auto()
    TO_DECOMPRESS = auto()
    TO_OV = auto()

STRIP_MODE = StripMode.TO_FLOAT
MODE = nncf.CompressWeightsMode.INT4_ASYM
BACKUP_MODE = nncf.BackupMode.INT8_ASYM
TORCH_DTYPE = torch.float32
# TORCH_DTYPE = torch.bfloat16
# TODO: error on export
#   attn_output = torch.nn.functional.scaled_dot_product_attention(
#   RuntimeError: Expected query, key, and value to have the same dtype, but got query.dtype: c10::BFloat16 key.dtype: float and value.dtype: float instead.
# TODO: support symmetric with signed scale.
# MODE = nncf.CompressWeightsMode.INT4_SYM
# BACKUP_MODE = nncf.BackupMode.INT8_SYM
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
GROUP_SIZE = 32
NUM_EVAL_SAMPLES = 1
WWB_REF = 'ref_wwb.csv'
assert Path(WWB_REF).exists()
EXP_NAME = f'mode={MODE.value}__backup={BACKUP_MODE.value}'
CKPT_PATH = Path(EXP_NAME + '.pth')
OV_DIR = Path(EXP_NAME + '_export')
OV_DIR.mkdir(exist_ok=True, parents=True)

def save_checkpoint(wrapped_model, ckpt_path='nncf_checkpoint.pth'):
    wrapped_model = wrapped_model.cpu()
    nncf_state_dict = wrapped_model.nncf.state_dict()
    nncf_config = wrapped_model.nncf.get_config()
    print(f"Saving ckpt to: {ckpt_path}")
    torch.save(
        {
            "nncf_state_dict": nncf_state_dict,
            "nncf_config": nncf_config,
        },
        ckpt_path,
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
    trust_remote_code=True,
).cuda()

tokenized_text = tokenizer("example" * 10, return_tensors="pt")
labels = tokenized_text["input_ids"].cuda()
attention_mask = tokenized_text["attention_mask"].cuda()
input_ids = labels[:, :-1]
labels = labels[:, 1:]
position_ids = torch.cumsum(attention_mask, axis=1).cuda() - 1
position_ids[attention_mask == 0] = 1
dataset = [{"input_ids": input_ids, "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}]


if CKPT_PATH.exists():
    nncf_ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=dataset[0])
    model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
    model = model.cuda()
else:
    nncf.compress_weights(
        model,
        ratio=1,
        group_size=GROUP_SIZE,
        mode=MODE,
        backup_mode=BACKUP_MODE,
        dataset=nncf.Dataset(dataset),
    )
    save_checkpoint(model, CKPT_PATH)

if STRIP_MODE != StripMode.NONE:
    if STRIP_MODE == StripMode.TO_FLOAT:
        for name, quantizer in model._nncf.external_quantizers.items():
            layer = get_module_by_name(quantizer.module_name, model)
            FQ_W = quantizer.quantize(layer.weight)
            layer.weight = torch.nn.Parameter(FQ_W)
        model._nncf.external_quantizers = None
        ctx = model._nncf.get_tracing_context()
        ctx.disable_tracing()
        ctx._post_hooks = {}
        ctx._pre_hooks = {}
    elif STRIP_MODE in [StripMode.TO_DECOMPRESS, StripMode.TO_OV]:
        from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model
        model = strip_tuned_lora_model(model)
        if STRIP_MODE == StripMode.TO_OV:
            if not (OV_DIR / 'openvino_model.bin').exists():
                # TODO: without it export fails with cuda:0 vs cpu on embedding
                model = model.cpu()
                export_from_model(model, OV_DIR, stateful=False, compression_option="bf16")
            model = OVModelForCausalLM.from_pretrained(
                model_id=OV_DIR,
                trust_remote_code=True,
                load_in_8bit=False,
                compile=True,
                ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
            )

wwb_eval = TextEvaluator(tokenizer=tokenizer, gt_data=WWB_REF, test_data=str(WWB_REF), use_chat_template=True, num_samples=NUM_EVAL_SAMPLES, language='cn')
_, all_metrics = wwb_eval.score(model)
print("Similarity: ", float(all_metrics["similarity"].iloc[0]))