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

# from functools import partial

from pathlib import Path

import matplotlib.pyplot as plt
import torch

# from optimum.intel.openvino import OVConfig
# from optimum.intel.openvino import OVQuantizer
# from peft import LoraConfig
# from peft import TaskType
# from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf

# from nncf.common.utils.debug import nncf_debug
# from nncf.quantization.advanced_parameters import QuantizationParameters

# model_id = "facebook/opt-125m"
model_id = "TinyLlama/TinyLlama_v1.1"

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",  # torch.float32,  # "auto",
    device_map="auto",
    low_cpu_mem_usage=True,
)  # .to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# print(hf_model)

output = hf_model.generate(
    tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=32, max_new_tokens=32
)
print("#" * 50 + " Before\n", tokenizer.decode(output[0]), "\n" + "#" * 150)

# Add LoRA
# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["layers.11.self_attn.v_proj"],
# )
# hf_model = get_peft_model(hf_model, peft_config)
# hf_model.print_trainable_parameters()


# We'll teach the model to repeatedly say "chicken".
tokenized_text = tokenizer("chicken " * 10, return_tensors="pt")
labels = tokenized_text["input_ids"].cuda()  # to("cuda:0")
attention_mask = tokenized_text["attention_mask"].cuda()  # to("cuda:0")
input_ids = labels[:, :-1]
labels = labels[:, 1:]

# Quantize with NNCF

# dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False, stateful=False)

# def transform_fn(tokenized_text):
#     input_ids = tokenized_text["input_ids"]
#     attention_mask = tokenized_text["attention_mask"]

#     inputs = {}
#     inputs["input_ids"] = input_ids
#     inputs["attention_mask"] = tokenized_text["attention_mask"]
#     position_ids = np.cumsum(attention_mask, axis=1) - 1
#     position_ids[attention_mask == 0] = 1

#     inputs["position_ids"] = position_ids
#     return inputs

position_ids = torch.cumsum(attention_mask, axis=1) - 1
position_ids[attention_mask == 0] = 1

dataset = [
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask[:, :-1],
        # "position_ids": position_ids
    }
]


# params = nncf.AdvancedQuantizationParameters(
#     disable_channel_alignment = True,
#     disable_bias_correction = True,
#     weights_quantization_params = QuantizationParameters(
#         num_bits=4,
#     ),
# )

# # Quantize weights only
# quantization_config = {
#     "algorithm": "quantization",
#     "weights": {
#         "mode": "symmetric",
#         "per_channel": True,
#         "bits": 4,
#         "target_scopes": [
#             "{re}.*OPTDecoderLayer\[11\]/OPTAttention\[self_attn\]/NNCFLinear\[v_proj\].*"
#         ],
#     },
#     "activations": {
#         "ignored_scopes": [
#             "{re}.*__add___.*",
#             "{re}.*__radd___.*",
#             "{re}.*layer_norm_.*",
#             "{re}.*__truediv__*",
#             "{re}.*__mul___.*",
#             "{re}.*bmm.*",
#             "{re}.*__rmul___.*",
#             "{re}.*tanh_.*",
#             "{re}.*pow_.*",
#             "{re}.*matmul.*",
#             "{re}.*addmm.*",
#             "{re}.*baddmm.*",
#             "{re}.*linear*",
#         ]
#     },
# }
# config = OVConfig(compression=quantization_config)
# config.target_device = "TRIAL"
# tokenizer.pad_token = tokenizer.eos_token


# def preprocess_fn(examples, tokenizer):
#     return tokenizer(examples["sentence"], padding=True, truncation=True, max_length=256)


# quantizer = OVQuantizer.from_pretrained(hf_model)
# calibration_dataset = quantizer.get_calibration_dataset(
#     "glue",
#     dataset_config_name="sst2",
#     preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
#     num_samples=2,
#     dataset_split="train",
#     preprocess_batch=True,
# )

# # TODO: for some reason no samples collected for weight statistics
# #   File "/home/nlyaly/projects/nncf/nncf/torch/tensor_statistics/collectors.py", line 46, in <lambda>
# # return map(lambda stat: stat.reshape(target_shape), targets)
# # AttributeError: 'NoneType' object has no attribute 'reshape
# quantizer.quantize(
#     calibration_dataset=calibration_dataset, save_directory="quantized_model", quantization_config=config
# )

# nncf.quantize(
#     hf_model.model,
#     target_device=nncf.TargetDevice.ANY,
#     ignored_scope=nncf.IgnoredScope(
#         # patterns = ['^(?!model.decoder.layers\[11\]\.v_proj$).*']
#         patterns=[
#             '^(?!.*OPTDecoderLayer\[11\]/OPTAttention\[self_attn\]/NNCFLinear\[v_proj\].*)'
#         ]
#         # patterns = ['.*_proj.*', '.*out_proj.*', '.*q_proj.*', '.*fc1.*', '.*fc2.*']
#     ),
#     calibration_dataset=nncf.Dataset(dataset),
#     advanced_parameters=params
# )


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


model = hf_model.model
nncf.compress_weights(
    hf_model.model,
    mode=nncf.CompressWeightsMode.INT8_ASYM,
    ignored_scope=nncf.IgnoredScope(
        #     # patterns=[
        #     #     # '^(?!model.decoder.layers\[11\]\.v_proj$).*'
        #     #             # '^(?!.*OPTDecoderLayer\[11\]/OPTAttention\[self_attn\]/NNCFLinear\[v_proj\]).*'
        #     #             # "^(?!.*OPTDecoderLayer\[5\]\/OPTAttention\[self_attn\]\/Linear\[v_proj\]\/l.*$).*"
        #     #             # "^(?!.*OPTDecoderLayer\[5\]\/OPTAttention\[self_attn\]\/Linear\[v_proj\]\/l.*$).*"
        # "^(?!.*LlamaModel\/ModuleList\[layers\]\/LlamaDecoderLayer\[21\]
        # \/LlamaSdpaAttention\[self_attn\]\/Linear\[v_proj\].*$).*"
        #     # ]
        #     #     # OPTDecoderLayer[11]/OPTAttention[self_attn]/Linear[v_proj]/to_0
        patterns=[
            #             # '.*_proj.*', '.*out_proj.*', '.*q_proj.*', '.*fc1.*',
            #             # '.*self_attn.*',
            #             '.*down_proj.*',
            #             '.*gate_proj.*', '.*up_proj.*',
            ".*embed_tokens.*"
        ]
    ),
    dataset=nncf.Dataset(dataset),
)
# model.nncf.get_graph().visualize_graph("fq_model.dot")
# print(hf_model.model)
hf_model.save_pretrained("/home/nlyaly/MODEL_DIR/tiny_llama_v1.1/fq4_emb32")
tokenizer.save_pretrained("/home/nlyaly/MODEL_DIR/tiny_llama_v1.1/fq4_emb32")

output = hf_model.generate(
    tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=32, max_new_tokens=32
)
print("#" * 50 + " After Quantize\n", tokenizer.decode(output[0]), "\n" + "#" * 150)

# TODO: select only adapters, disable grad even for quantization params

# def set_requires_grad_true(module):
#     if hasattr(module, 'weight') and module.weight is not None:
#         module.weight.requires_grad = False
#     if hasattr(module, 'bias') and module.bias is not None:
#         module.bias.requires_grad = True
# hf_model.model.apply(set_requires_grad_true)


for param in hf_model.parameters():
    param.requires_grad = False

param_to_train = []
for name, param in hf_model.named_parameters():
    if "lora" in name:  # or "11.self_attn.v_proj.weight" in name:  # or 'input' in name:
        # print("optimize -> ", name)
        param.requires_grad = True
        param_to_train.append(param)
    # if "11.self_attn.v_proj.weight" in name:
    #     param.requires_grad = True
    #     param_to_train.append(param)
    # if 'input' in name:
    #     print('require grad -> ', name)
    #     param.requires_grad = True

for name, param in hf_model.named_parameters():
    if param.requires_grad:
        print("requires grad for -> ", name)
print_trainable_parameters(model)

optimizer = torch.optim.Adam(hf_model.parameters(), lr=1e-4)
losses = []
for i in range(50):
    optimizer.zero_grad()
    loss = hf_model(input_ids=input_ids, labels=labels).loss
    losses.append(float(loss))
    # print(float(loss))
    loss.backward()
    optimizer.step()

hf_model.save_pretrained("/home/nlyaly/MODEL_DIR/tiny_llama_v1.1/fq4_emb32_chicken")
tokenizer.save_pretrained("/home/nlyaly/MODEL_DIR/tiny_llama_v1.1/fq4_emb32_chicken")

# print("weight after ", model.layers[21].self_attn.v_proj.weight[:5, :5])
# print('weight before ', weight[:5, :5])
# quantizer = model.nncf.external_quantizers.FQ_LORA_for_node_layers_21_self_attn_v_proj_weight
# for name, param in quantizer.named_parameters():
#     print(name, param[:5])

# Check that loss is decreasing
plt.plot(losses)
plt.title("Lora fine-tuning", fontsize=20)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
path = Path("loss.png").resolve()
plt.savefig(path)
print("Saving loss plot to:", path)

# Check the output of tuned model
output = tokenizer.decode(
    hf_model.generate(
        tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=32, max_new_tokens=32
    )[0]
)
print("#" * 50 + " After Tune\n", output, "\n" + "#" * 150)
print(f"Peak memory usage: {torch.cuda.max_memory_allocated() * 1e-9:.2f} Gb")