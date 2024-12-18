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
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch import load_from_config


def load_nncf_ckpt(model, tokenizer):
    tokenized_text = tokenizer("example", return_tensors="pt")
    input_ids = tokenized_text["input_ids"]
    attention_mask = tokenized_text["attention_mask"]
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    example_input = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "position_ids": position_ids.cuda(),
    }
    path = '/local_ssd2/nlyalyus/MODEL_DIR/SmolLM-1_7B-Instruct/FQ_emb_head_bf16_int4_asym_rank256_gs64/nncf_checkpoint.pth'
    nncf_ckpt = torch.load(path)
    model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=example_input)
    model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
    return model

def strip_model(model, strip_int8=True, strip_int4=True):
    for name, quantizer in model._nncf.external_quantizers.items():
        if quantizer.levels == 16 and not strip_int4 or quantizer.levels == 256 and not strip_int8:
            continue
        module_name = quantizer.module_name
        layer = get_module_by_name(module_name, model)
        W = layer.weight
        FQ_W = quantizer.quantize(W)

        diff = (W - FQ_W).t()
        print('diff={} for module={}'.format(torch.linalg.norm(diff, ord="fro").item(), module_name))
        layer.weight = torch.nn.Parameter(FQ_W)
    model._nncf.external_quantizers = None
    ctx = model._nncf.get_tracing_context()
    print('num post-hooks={} pre_hooks={}'.format(len(ctx._post_hooks), len(ctx._pre_hooks)))
    ctx._post_hooks = {}
    ctx._pre_hooks = {}
    return model

def ask_model(model, tokenizer):
    device = next(iter(model.parameters())).device
    messages = [{"role": "user", "content": "What is the capital of France."}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, do_sample=False)
    # outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    print(tokenizer.decode(outputs[0]))


MODEL_ID = "HuggingFaceTB/SmolLM-1.7B-Instruct"
student_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
ask_model(student_model, tokenizer)

int4_model = load_nncf_ckpt(student_model, tokenizer)
int4_model = strip_model(int4_model, strip_int8=True, strip_int4=True)
ask_model(int4_model, tokenizer)
