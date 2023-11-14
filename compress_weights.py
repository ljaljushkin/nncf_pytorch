import atexit
import datetime
import gc
import json
import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import openvino.runtime as ov
from datasets import load_dataset
from openvino import Core
from optimum.intel import OVModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

from nncf import Dataset
from nncf import compress_weights
from nncf.parameters import CompressWeightsMode

core = Core()

def gen_pkv(num_heads, head_dim, num_layers=None):
    if num_layers is None:
        num_layers = num_heads
    res = {}
    for i in range(num_layers):
        res[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, head_dim))
        res[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, head_dim))
    return res

def gen_pkv_bloom(num_heads, head_dim, num_layers=None):
    if num_layers is None:
        num_layers = num_heads
    res = {}
    for i in range(num_layers):
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        res[f"past_key_values.{i}.key"] = np.zeros((1 * num_heads, head_dim, 0))
        res[f"past_key_values.{i}.value"] = np.zeros((1 * num_heads, 0, head_dim))
    return res

def transform_func(item, tokenizer, gen_pkv_fn):
    tokens = tokenizer(item['text'])
    #return tokens['input_ids'], tokens['attention_mask']
    attention_mask = np.expand_dims(np.array(tokens['attention_mask']), 0)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = np.ma.array(position_ids, mask=attention_mask == 0)
    position_ids.filled(fill_value=1)
    res = {
        'input_ids': np.expand_dims(np.array(tokens['input_ids']), 0),
        'attention_mask': attention_mask,
        # 'position_ids': position_ids
    }
    res.update(gen_pkv_fn())
    return res

MODEL_IDS_VS_GEN_FN = {
    'facebook/opt-125m': partial(gen_pkv, 12, 64),
    'databricks/dolly-v2-3b': partial(gen_pkv, 32, 80),
    'meta-llama/Llama-2-7b-chat-hf': partial(gen_pkv, 32, 128),
    'facebook/opt-6.7b': partial(gen_pkv, 32, 128),
    'bigscience/bloom-7b1': partial(gen_pkv_bloom, 32, 128, 30),
    'togethercomputer/RedPajama-INCITE-7B-Instruct': partial(gen_pkv, 32, 128),
    'meta-llama/Llama-2-13b-chat-hf': partial(gen_pkv, 40, 128),
    'databricks/dolly-v2-12b': partial(gen_pkv, 40, 128, 36),
    'openlm-research/open_llama_3b': None,
    'THUDM/chatglm2-6b': None,
    'HuggingFaceH4/zephyr-7b-beta': None,
}


@dataclass
class ExpDesc:
    model_id: str
    mode: CompressWeightsMode = CompressWeightsMode.INT4_SYM
    ratio: int = 1
    group_size: int = 128
    is_revert: bool = False
    is_data: bool = False

    def __str__(self):
        return f'{self.model_id} ----> {self.get_exp_name()}'

    def get_compress_fn(self):
        nncf_dataset = None
        if self.is_data:
            gen_pkv_fn = MODEL_IDS_VS_GEN_FN[self.model_id]
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train[:1000]')
            dataset = dataset.filter(lambda example: len(example["text"]) > 128)
            nncf_dataset = Dataset(dataset, partial(transform_func, tokenizer=tokenizer, gen_pkv_fn=gen_pkv_fn))

        return partial(compress_weights, mode=self.mode, ratio=self.ratio, group_size=self.group_size, dataset=nncf_dataset, is_revert=self.is_revert)

    def get_exp_name(self):
        result = "int4"
        if self.mode == CompressWeightsMode.INT8:
            result = "int8"
        elif self.mode == CompressWeightsMode.NF4:
            result = "nf4"

        if self.group_size != -1:
            result += f'_g{self.group_size}'

        if self.mode == CompressWeightsMode.INT4_SYM:
            result += '_nozp'

        if self.ratio != 1:
            result += f'_r{self.ratio * 100:2.0f}'

        if self.is_data:
            if self.is_revert:
                result += '_anti'
            else:
                result += '_criteria'
        else:
            if self.is_revert:
                result += '_a0'
        return result

EXP_DESCS= [
    # ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.NF4, ratio=0.5, group_size=64),
    ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_SYM),
    # ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128),
    # ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, is_revert=True),
    # ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, is_data=True),
    # ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, is_data=True, is_revert=True),

    # ExpDesc('databricks/dolly-v2-3b', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128),
    # ExpDesc('databricks/dolly-v2-3b', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, is_revert=True)),
    # ExpDesc('databricks/dolly-v2-3b', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, is_data=True),
    # ExpDesc('databricks/dolly-v2-3b', mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, is_data=True, is_revert=True),
]

# EXP_DESCS = [ExpDesc(model_id, fn, name) for model_id in MODEL_IDS for fn, name in MODES_AND_NAMES]

is_bin_needed = True
cache_dir = Path('cache')
ov_name = 'openvino_model.xml'

print('All experiments summary:')
for desc in EXP_DESCS:
    print(desc)

for desc in tqdm(EXP_DESCS):
    print(desc)
    model_id = desc.model_id
    exp_name = desc.get_exp_name()
    model_name = Path(model_id).name
    gen_pkv_fn = MODEL_IDS_VS_GEN_FN[model_id]

    SRC_PATH = cache_dir / model_name / 'fp32'/  ov_name
    print(SRC_PATH)
    try:
        if not SRC_PATH.with_suffix('.bin').exists():
            use_pkv = True
            ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, export=True, from_transformers=True)
            ov_model.save_pretrained(SRC_PATH.parent)
            ov_model._save_config(SRC_PATH.parent)
            fp32_model = ov_model.model
        else:
            fp32_model = core.read_model(model=SRC_PATH)
    except Exception as error:
        print("Reading FP32 model failed:", error)
        continue

    print(f'\n\n{model_id}___{exp_name}')

    DST_PATH = cache_dir / model_name / exp_name /  ov_name
    DST_PATH.parent.mkdir(exist_ok=True)
    print(DST_PATH)
    shutil.copyfile(SRC_PATH.parent / 'config.json', DST_PATH.parent / 'config.json')

    try:
        start = time.time()
        model = desc.get_compress_fn()(fp32_model)
        print(f'compressing weights took {(time.time() - start):.1f} seconds')
    except Exception as error:
        print("Compression failed:", error)
        print(traceback.print_exc())
        continue

    start = time.time()
    ov.save_model(model, DST_PATH, compress_to_fp16=False)
    print(f"saving model {DST_PATH} took {(time.time() - start):.1f} seconds")

    if not is_bin_needed:
        file_to_remove = DST_PATH.rename(DST_PATH.with_suffix('.bin'))
        Path.unlink(file_to_remove)

    del model
    gc.collect()