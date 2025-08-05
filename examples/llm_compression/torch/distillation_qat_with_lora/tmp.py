from pprint import pprint

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

pretrained_models = [
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "tinyllama/tinyllama-1.1b-step-50k-105b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "bigcode/starcoder2-3b",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "stabilityai/stablelm-3b-4e1t",
]

d = {}
for pretrained in pretrained_models:
    print(f"\nAnalyzing model: {pretrained}")
    try:
        model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="auto")

        s = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                s.add(tuple(module.weight.shape))
        # print(f"Linear layer shapes: {s}")
        d[pretrained] = s
        # Free up memory
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading {pretrained}: {e}")

pprint(d)
# Collect all unique shapes from all models
ss = set()
for shapes in d.values():
    ss.update(shapes)
print(f"All linear layer shapes: {list(ss)}")
