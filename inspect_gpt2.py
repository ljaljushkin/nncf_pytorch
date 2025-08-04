#!/usr/bin/env python3
"""Quick script to inspect GPT-2 model structure"""

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
print("All parameter names:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

print("\nAll module names:")
for name, module in model.named_modules():
    if hasattr(module, "weight"):
        print(f"  {name}: {type(module)} - weight shape: {module.weight.shape}")

print(f"\nModel config: {model.config}")
print(f"Model tie_word_embeddings: {getattr(model.config, 'tie_word_embeddings', 'Not found')}")
