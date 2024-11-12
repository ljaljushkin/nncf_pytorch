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

import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf


def generate_overfit(pipeline, tokenizer, prefix=""):
    output = pipeline.generate(
        tokenizer("overfit", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=32, max_new_tokens=32
    )
    print("#" * 50 + f" {prefix}\n", tokenizer.decode(output[0]), "\n" + "#" * 150)


def get_nb_trainable_parameters(module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in module.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(module):
    trainable_params, all_param = get_nb_trainable_parameters(module)

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def save_checkpoint(wrapped_model, ckpt_dir):
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    wrapped_model = wrapped_model.cpu()
    nncf_state_dict = wrapped_model.nncf.state_dict()
    nncf_config = wrapped_model.nncf.get_config()
    torch.save(
        {
            "nncf_state_dict": nncf_state_dict,
            "nncf_config": nncf_config,
        },
        ckpt_dir / "nncf_checkpoint.pth",
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

ROOT_MODEL_DIR = Path.home() / ("MODEL_DIR")

model_id = "microsoft/Phi-3-mini-4k-instruct"
model_name = Path(model_id).name.replace(".", "_")

MODEL_DIR = ROOT_MODEL_DIR / model_name

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # torch.float32, # "auto",  # torch.float32,  # "auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
generate_overfit(hf_model, tokenizer, "FP32")

# We'll teach the model to repeatedly say "overfit".
tokenized_text = tokenizer("overfit " * 10, return_tensors="pt")
labels = tokenized_text["input_ids"].cuda()  # to("cuda:0")
attention_mask = tokenized_text["attention_mask"].cuda()  # to("cuda:0")
input_ids = labels[:, :-1]
labels = labels[:, 1:]
position_ids = torch.cumsum(attention_mask, axis=1) - 1
position_ids[attention_mask == 0] = 1

dataset = [{"input_ids": input_ids, "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}]


model = hf_model.model
nncf.compress_weights(
    hf_model.model,
    mode=nncf.CompressWeightsMode.INT8_ASYM,
    ignored_scope=nncf.IgnoredScope(
        # patterns=[
        #     # #    #     # '^(?!model.decoder.layers\[11\]\.v_proj$).*'
        #     # #    #             # '^(?!.*OPTDecoderLayer\[11\]/OPTAttention\[self_attn\]/NNCFLinear\[v_proj\]).*'
        #     # #    #             # "^(?!.*OPTDecoderLayer\[5\]\/OPTAttention\[self_attn\]\/Linear\[v_proj\]\/l.*$).*"
        #     # #    #             # "^(?!.*OPTDecoderLayer\[5\]\/OPTAttention\[self_attn\]\/Linear\[v_proj\]\/l.*$).*"
        #     # #    "^(?!.*LlamaModel\/ModuleList\[layers\]\/LlamaDecoderLayer\[21\].*$).*"
        #     # #        # \/LlamaSdpaAttention\[self_attn\]\/Linear\[v_proj\].*$).*"
        #     # # # OPTDecoderLayer[11]/OPTAttention[self_attn]/Linear[v_proj]/to_0
        #     "^(?!.*Phi3DecoderLayer\[31\].*$).*"
        # ]
        patterns=[
            # #     #     #             y# '.*_proj.*', '.*out_proj.*', '.*q_proj.*', '.*fc1.*',
            # #     #     #             # '.*self_attn.*',
            # #     #     #             '.*down_proj.*',
            # #     #     #             '.*gate_proj.*', '.*up_proj.*',
            ".*embed_tokens.*"
        ]
    ),
    dataset=nncf.Dataset(dataset),
)

generate_overfit(hf_model, tokenizer, "Quantized")
ckpt_dir = MODEL_DIR / "FQ_4bit_no_embed_svd_rank256_g64_bloat16"
# ckpt_dir = MODEL_DIR / "FQ_4bit_no_embed_svd_rank8"
save_checkpoint(hf_model.model, ckpt_dir)
model.nncf.get_graph().visualize_graph(ckpt_dir / "fq_model.dot")
