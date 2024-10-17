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

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model_id = "facebook/opt-125m"

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(hf_model)

output = hf_model.generate(
    tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=128, max_new_tokens=128
)
print("#" * 50 + " Before\n", tokenizer.decode(output[0]), "\n" + "#" * 150)

# Add LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["layers.11.self_attn.v_proj"],
)
hf_model = get_peft_model(hf_model, peft_config)
hf_model.print_trainable_parameters()


# We'll teach the model to repeatedly say "chicken".
labels = tokenizer("chicken " * 10, return_tensors="pt")["input_ids"].to("cuda:0")
input_ids = labels[:, :-1]
labels = labels[:, 1:]

optimizer = torch.optim.Adam(hf_model.parameters(), lr=1e-2)

losses = []
for i in range(10):
    optimizer.zero_grad()
    loss = hf_model(input_ids=input_ids, labels=labels).loss
    losses.append(float(loss))
    loss.backward()
    optimizer.step()

# Check that loss is decreasing
plt.plot(losses)
plt.title("Lora fine-tuning", fontsize=20)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()

# Check the output of tuned model
output = tokenizer.decode(hf_model.generate(tokenizer("chicken", return_tensors="pt")["input_ids"].cuda())[0])
print("#" * 50 + " After\n", output, "\n" + "#" * 150)
print(f"Peak memory usage: {torch.cuda.max_memory_allocated() * 1e-9:.2f} Gb")
