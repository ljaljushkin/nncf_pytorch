#!/usr/bin/env python3
"""
Simple script to help discover layer names in a model for comparison.
This helps you find which layers to analyze with the compare_model_outputs.py tool.
"""

import argparse

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def list_model_layers(model_id: str, max_layers: int = 50):
    """List all layer names in a model to help choose layers for comparison."""
    print(f"Loading model: {model_id}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")

    print(f"\nModel architecture: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nAll layer names (showing first {max_layers}):")
    print("-" * 60)

    layer_count = 0
    layer_types = {}

    for name, module in model.named_modules():
        if layer_count < max_layers:
            module_type = type(module).__name__
            print(f"{name:<40} | {module_type}")

            # Count layer types
            if module_type in layer_types:
                layer_types[module_type] += 1
            else:
                layer_types[module_type] = 1

        layer_count += 1

    if layer_count > max_layers:
        print(f"... and {layer_count - max_layers} more layers")

    print(f"\nLayer type summary:")
    print("-" * 40)
    for layer_type, count in sorted(layer_types.items()):
        print(f"{layer_type:<30} | {count:>3}")

    print(f"\nCommon layer patterns to try:")
    print("-" * 40)

    # Suggest common patterns based on layer names
    suggestions = []
    layer_names = [name for name, _ in model.named_modules()]

    if any("embed" in name.lower() for name in layer_names):
        embed_layers = [name for name in layer_names if "embed" in name.lower()]
        suggestions.append(f"Embedding layers: {embed_layers[:3]}")

    if any("lm_head" in name.lower() for name in layer_names):
        suggestions.append("Language model head: ['lm_head']")

    if any("attention" in name.lower() for name in layer_names):
        attn_layers = [name for name in layer_names if "attention" in name.lower()][:3]
        suggestions.append(f"Attention layers: {attn_layers}")

    if any("mlp" in name.lower() for name in layer_names):
        mlp_layers = [name for name in layer_names if "mlp" in name.lower()][:3]
        suggestions.append(f"MLP layers: {mlp_layers}")

    # Look for transformer layers
    transformer_layers = [
        name for name in layer_names if any(x in name.lower() for x in ["layer", "block", "transformer"])
    ][:3]
    if transformer_layers:
        suggestions.append(f"Transformer layers: {transformer_layers}")

    for suggestion in suggestions:
        print(f"  {suggestion}")


def main():
    parser = argparse.ArgumentParser(description="Discover layer names in a model")
    parser.add_argument(
        "--model-id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face model ID"
    )
    parser.add_argument("--max-layers", type=int, default=50, help="Maximum number of layers to display")

    args = parser.parse_args()

    try:
        list_model_layers(args.model_id, args.max_layers)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
