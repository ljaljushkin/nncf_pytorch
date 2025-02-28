# Copyright (c) 2025 Intel Corporation
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
import json
from pathlib import Path

# from optimum.exporters.openvino.convert import export_from_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import TextEvaluator

import torch
from nncf.torch import load_from_config
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name

parser = argparse.ArgumentParser(add_help=True)
# Model params
parser.add_argument("-m", "--model_id")
parser.add_argument("-n", "--nncf_ckpt_dir")
args = parser.parse_args()

model_name = Path(args.model_id).name.replace(".", "_")
model_dir = Path.home() / "MODEL_DIR" / model_name

tokenizer = AutoTokenizer.from_pretrained(
    args.model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # TODO: doesn't work with strip
    #   attn_output = torch.nn.functional.scaled_dot_product_attention(
    #   RuntimeError: Expected query, key, and value to have the same dtype, but got query.dtype:
    #  c10::BFloat16 key.dtype: float and value.dtype: float instead.
).cuda()

# chat_template = [{"role": "user", "content": "input_text"}]
wwb_ref = model_dir / "wwb_torch_ref_chat.csv"
# wwb_eval = None
# if wwb_ref.exists():
#     print("Loading cached WWB reference answers from: ", wwb_ref.resolve())
wwb_eval = TextEvaluator(
    tokenizer=tokenizer, gt_data=wwb_ref, test_data=str(wwb_ref), use_chat_template=True, language="en"
)
# else:
# wwb_eval = TextEvaluator(
#     base_model=model, tokenizer=tokenizer, use_chat_template=True, metrics=("similarity",), language='en'
# )
# wwb_eval.dump_gt(str(wwb_ref))


nncf_ckpt_dir = Path(args.nncf_ckpt_dir)


tokenized_text = tokenizer("chicken " * 10, return_tensors="pt")
input_ids = tokenized_text["input_ids"].cuda()
attention_mask = tokenized_text["attention_mask"].cuda()
position_ids = (torch.cumsum(attention_mask, axis=1) - 1).cuda()
position_ids[attention_mask == 0] = 1

dataset = [
    {"input_ids": input_ids[:, :-1], "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}
]
nncf_ckpt = torch.load(nncf_ckpt_dir / "nncf_checkpoint.pth", map_location="cpu", weights_only=False)
# NOTE: assume that the whole hf_model=AutoModelForCausalLM(...) was passed to NNCF for compression
# TODO: won't work with accelerator, the model is not supposed to be overriden? see @property model in HFLM
model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=dataset[0])
model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
model.cuda()

float_strip = True
if float_strip:
    layout = model.nncf.transformation_layout()
    model = model.nncf.get_clean_shallow_copy()
    graph = model.nncf.get_graph()
    t = layout.transformations
    for command in t:
        quantizer = command.fn
        tp = command.target_points[0]
        node_with_weight = graph.get_node_by_name(tp.target_node_name)
        weight_node = get_const_node(node_with_weight, tp.input_port_id, graph)
        weight_name = weight_node.layer_attributes.name
        module_name, weight_attr_name = split_const_name(weight_name)
        layer = get_module_by_name(module_name, model)
        weight = getattr(layer, weight_attr_name)
        fq_weight = quantizer.quantize(layer.weight)
        setattr(layer, weight_attr_name, torch.nn.Parameter(fq_weight))
    # for name, quantizer in model._nncf.external_quantizers.items():
    #     layer = get_module_by_name(quantizer.module_name, model)
    #     FQ_W = quantizer.quantize(layer.weight)
    #     layer.weight = torch.nn.Parameter(FQ_W)
    model._nncf.external_quantizers = None
    ctx = model._nncf.get_tracing_context()
    ctx.disable_tracing()
    ctx._post_hooks = {}
    ctx._pre_hooks = {}
# else:
# from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model

# model = strip_tuned_lora_model(model)
# ov_dir = nncf_ckpt_dir / 'exported'
# ov_dir.mkdir(exist_ok=True, parents=True)
# model = model.cpu()  # cuda:0 vs cpu on embedding
# export_from_model(model, ov_dir, stateful=False, compression_option="bf16")

results_file = nncf_ckpt_dir / "results_wwb_chat.json"
all_metrics_per_question, all_metrics = wwb_eval.score(model)
similarity = float(all_metrics["similarity"].iloc[0])
results = {"results": {"WWB": {"similarity": similarity}}}
print(json.dumps(results, indent=4))
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
