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

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_creation import wrap_model
from nncf.torch.model_transformer import PTModelTransformer

# model_id = "facebook/opt-125m"
model_id = "TinyLlama/TinyLlama_v1.1"

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # "auto",
    # device_map="auto",
    # low_cpu_mem_usage=True,
)  # .to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)


class AdditiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, A, B):
        # Save the additive parameter for backward pass
        ctx.save_for_backward(A, B)
        # Perform the forward pass
        # print(W.shape)
        return W + B @ A

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad shape', grad_output.shape)
        # Retrieve the saved tensor
        A, B = ctx.saved_tensors
        # Compute the gradient for the additive parameter
        # grad_A = grad_output.clone()
        grad_A = B.t() @ grad_output  # Gradient of the loss w.r.t. A
        grad_B = grad_output @ A.t()  # Gradient of the loss w.r.t. B
        # No gradient for W since it is frozen
        grad_W = None
        return grad_W, grad_A, grad_B


class FQLora(nn.Module):
    def __init__(self):
        super().__init__()
        out_features, in_features = 256, 2048
        # out_features, in_features = 768, 768
        lora_rank = 8
        self._A = torch.nn.Parameter(
            torch.ones((lora_rank, in_features), dtype=torch.float32), requires_grad=True
        )  # [L, I]
        self._B = torch.nn.Parameter(
            torch.zeros((out_features, lora_rank), dtype=torch.float32), requires_grad=True
        )  # [O, L]
        print(self._A.shape)
        print(self._B.shape)

    def forward(self, weight):
        # return weight + self._B @ self._A
        # weight = weight.detach()
        # for name, param in self.named_parameters():
        #     print("CHECK: ", name, param.requires_grad, param.shape)
        # print("CHECK: weight ", weight.requires_grad, weight.shape)
        return AdditiveFunction.apply(weight, self._A, self._B)


model = hf_model.model
tokenized_text = tokenizer("chicken " * 10, return_tensors="pt")
labels = tokenized_text["input_ids"]  # .to("cuda:0")
attention_mask = tokenized_text["attention_mask"]  # .to("cuda:0")
input_ids = labels[:, :-1]
labels = labels[:, 1:]
dataset = [
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask[:, :-1],
        # "position_ids": position_ids
    }
]
output = hf_model.generate(
    tokenizer("chicken", return_tensors="pt")["input_ids"], min_new_tokens=32, max_new_tokens=32  # .cuda(),
)
print("#" * 50 + " Before Quantize\n", tokenizer.decode(output[0]), "\n" + "#" * 150)


model = wrap_model(model, example_input=dataset[0], trace_parameters=True)
# print(model)

transformation_layout = TransformationLayout()
quantizer = FQLora()
# "OPTModel/OPTDecoder[decoder]/ModuleList[layers]/OPTDecoderLayer[0]/OPTAttention[self_attn]/Linear[v_proj]/linear_0"
node_name = "LlamaModel/ModuleList[layers]/LlamaDecoderLayer[21]/LlamaSdpaAttention[self_attn]/Linear[v_proj]/linear_0"
target_point = PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, node_name, input_port_id=1)
transformation_layout.register(
    PTSharedFnInsertionCommand(
        target_points=[target_point],
        fn=quantizer,
        op_unique_name="FQ_LORA_for_node_",
        compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
        priority=TransformationPriority.QUANTIZATION_PRIORITY,
    )
)
transformed_model = PTModelTransformer(model).transform(transformation_layout)
# print(transformed_model)
model.nncf.get_graph().visualize_graph("fq_model.dot")


hf_model.requires_grad_(False)
# for param in hf_model.parameters():
#     param.requires_grad = False

param_to_train = []
for name, param in hf_model.named_parameters():
    if "_A" in name or "_B" in name:  # or "11.self_attn.v_proj.weight" in name:  # or 'input' in name:
        param.requires_grad = True
        param_to_train.append(param)
        print("optimize -->", name)
    else:
        param.requires_grad = False


# hf_model.disable_input_require_grads()
# hf_model.enable_input_require_grads() # no error, but no gradient for adapters as well
# NOTE: Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
# the model weights fixed.
# NOTE: When training with PEFT, only LoRA layers will have requires grad set to True,
# but the output of frozen layers need to propagate the gradients to make sure the gradient flows.
# def make_inputs_require_grad(module, input, output):
#     output.requires_grad_(True)
# hf_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
# hf_model.get_input_embeddings().weight.requires_grad = True
# hf_model.lm_head.weight.requires_grad = True
print("embedding: ", hf_model.get_input_embeddings().weight.requires_grad)
print("lm_head: ", hf_model.lm_head.weight.requires_grad)

for name, param in hf_model.named_parameters():
    if param.requires_grad:
        print("requires grad for -> ", name)

# torch.set_grad_enabled(True)
# print(hf_model.get_input_embeddings().weight)
optimizer = torch.optim.Adam(param_to_train, lr=1e-2)
losses = []
for i in range(10):
    optimizer.zero_grad()
    loss = hf_model(input_ids=input_ids, labels=labels).loss
    losses.append(float(loss))
    loss.backward()
    # print(float(loss))
    optimizer.step()


plt.plot(losses)
plt.title("Lora fine-tuning", fontsize=20)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
path = Path("loss.png").resolve()
plt.savefig(path)
print("Saving loss plot to:", path)
# print(hf_model.get_input_embeddings().weight)

hf_model.requires_grad_(False)
with torch.inference_mode():
    # Check the output of tuned model
    output = tokenizer.decode(
        hf_model.generate(
            tokenizer("chicken", return_tensors="pt")["input_ids"],
            min_new_tokens=32,
            max_new_tokens=32,
            # .cuda()
        )[0]
    )
print("#" * 50 + " After Tune\n", output, "\n" + "#" * 150)
# print(f"Peak memory usage: {torch.cuda.max_memory_allocated() * 1e-9:.2f} Gb")

# Check the updated additive parameter
# print("Weights: ", model.linear.weight)
# print("Updated additive parameter A:", model.nncf.external_quantizers.FQ_LORA_for_node_._A)
# print("Updated additive parameter B:", model.nncf.external_quantizers.FQ_LORA_for_node_._B)
# print(output)
