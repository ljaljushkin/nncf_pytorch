#!/usr/bin/env python3
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

"""
Tool to compare outputs between original and compressed LLM models.
Finds top-k rows with largest differences for specified layers.
"""

import argparse
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf import CompressWeightsMode


class BackendType(Enum):
    """Supported model backends."""

    PYTORCH = "pytorch"
    OPENVINO = "openvino"


@dataclass
class LayerComparison:
    """Results of comparing a layer between original and compressed models."""

    layer_name: str
    top_k_indices: list[int]
    top_k_distances: list[float]
    mean_distance: float
    max_distance: float
    distance_metric: str


class ModelOutputExtractor:
    """Extracts outputs from specified layers during model inference."""

    def __init__(self, model, layer_names: list[str], backend: BackendType, calibration_dataset=None):
        self.model = model
        self.layer_names = layer_names
        self.backend = backend
        self.layer_outputs = {}
        self.hooks = []
        self.calibration_dataset = calibration_dataset
        self.collected_activations = {}

        if backend == BackendType.PYTORCH:
            self._register_pytorch_hooks()
        elif backend == BackendType.OPENVINO:
            self._setup_openvino_extraction()

    def _register_pytorch_hooks(self):
        """Register forward hooks for PyTorch models."""

        def create_hook(name):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, tuple):
                    # For layers that return multiple outputs, take the first one
                    self.layer_outputs[name] = output[0].detach().clone()
                else:
                    self.layer_outputs[name] = output.detach().clone()

            return hook

        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
                print(f"Registered hook for layer: {name}")

    def __call__(self, *args, **kwargs):
        """Forward pass with output extraction."""
        self.layer_outputs.clear()

        if self.backend == BackendType.PYTORCH:
            with torch.no_grad():
                output = self.model(*args, **kwargs)
            return output, self.layer_outputs.copy()

        elif self.backend == BackendType.OPENVINO:
            msg = "OpenVINO backend is not supported."
            raise RuntimeError(msg)

    def remove_hooks(self):
        """Remove all registered hooks (PyTorch only)."""
        if self.backend == BackendType.PYTORCH:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()


def load_model(model_id: str, backend: BackendType, device: str = "cpu"):
    """Load model based on the specified backend."""
    if backend == BackendType.PYTORCH:
        return load_pytorch_model(model_id, device)
    else:
        msg = f"Unsupported backend: {backend}"
        raise ValueError(msg)


def load_pytorch_model(model_id: str, device: str = "cpu"):
    """Load PyTorch model."""
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device)
    model.eval()
    return model


def compress_model(model, backend: BackendType, compression_params: dict, calibration_dataset):
    """Compress model based on backend type."""
    if backend == BackendType.PYTORCH:
        # Compress the model directly (no need to copy since we loaded it fresh)
        compressed_model = nncf.compress_weights(
            model,
            dataset=calibration_dataset,
            # ignored_scope=nncf.IgnoredScope(patterns=[".*embedding.*"]),
            ignored_scope=nncf.IgnoredScope(patterns=["(?!.*embed.*)\w+"]),
            **compression_params,
        )
        compressed_model.eval()
        return compressed_model
    else:
        msg = f"Unsupported backend for compression: {backend}"
        raise ValueError(msg)


def find_layer_names(model, target_patterns: list[str], backend: BackendType) -> list[str]:
    """Find layer names matching target patterns."""
    found_layers = []
    all_layers = []

    if backend == BackendType.PYTORCH:
        for name, module in model.named_modules():
            all_layers.append(name)
            for pattern in target_patterns:
                if pattern.lower() in name.lower():
                    found_layers.append(name)
                    break

    if not found_layers:
        print("No layers found matching patterns:", target_patterns)
        print("\nAvailable layers:")
        for layer in all_layers[:15]:  # Show first 15 layers
            print(f"  {layer}")
        if len(all_layers) > 15:
            print(f"  ... and {len(all_layers) - 30} more layers")
        for layer in all_layers[-15:]:  # Show last 15 layers
            print(f"  {layer}")

    return found_layers


def calculate_distance(orig_output: torch.Tensor, comp_output: torch.Tensor, metric: str = "l2") -> np.ndarray:
    """Calculate distance between original and compressed outputs."""
    # Convert to numpy and ensure same shape
    orig_np = orig_output.cpu().numpy()
    comp_np = comp_output.cpu().numpy()

    if orig_np.shape != comp_np.shape:
        msg = f"Shape mismatch: {orig_np.shape} vs {comp_np.shape}"
        raise ValueError(msg)

    # Calculate distance for each row (assuming shape is [batch, seq_len, hidden_dim])
    if len(orig_np.shape) == 3:
        # For each token position, calculate distance across hidden dimensions
        if metric == "l2":
            distances = np.linalg.norm(orig_np - comp_np, axis=2)  # Shape: [batch, seq_len]
        elif metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            orig_norm = np.linalg.norm(orig_np, axis=2, keepdims=True)
            comp_norm = np.linalg.norm(comp_np, axis=2, keepdims=True)
            cosine_sim = np.sum(orig_np * comp_np, axis=2) / (orig_norm.squeeze() * comp_norm.squeeze() + 1e-8)
            distances = 1 - cosine_sim
        elif metric == "mse":
            distances = np.mean((orig_np - comp_np) ** 2, axis=2)
        else:
            msg = f"Unknown metric: {metric}"
            raise ValueError(msg)
    elif len(orig_np.shape) == 2:
        # For 2D outputs, calculate distance across last dimension
        if metric == "l2":
            distances = np.linalg.norm(orig_np - comp_np, axis=1)
        elif metric == "cosine":
            orig_norm = np.linalg.norm(orig_np, axis=1, keepdims=True)
            comp_norm = np.linalg.norm(comp_np, axis=1, keepdims=True)
            cosine_sim = np.sum(orig_np * comp_np, axis=1) / (orig_norm.squeeze() * comp_norm.squeeze() + 1e-8)
            distances = 1 - cosine_sim
        elif metric == "mse":
            distances = np.mean((orig_np - comp_np) ** 2, axis=1)
        else:
            msg = f"Unknown metric: {metric}"
            raise ValueError(msg)
    else:
        msg = f"Unsupported tensor shape: {orig_np.shape}"
        raise ValueError(msg)

    return distances


def find_top_k_differences(
    orig_output: torch.Tensor, comp_output: torch.Tensor, k: int, metric: str = "l2"
) -> tuple[list[int], list[float]]:
    """Find top-k rows with largest differences."""
    distances = calculate_distance(orig_output, comp_output, metric)

    # Flatten distances if needed and get top-k
    flat_distances = distances.flatten()
    top_k_flat_indices = np.argsort(flat_distances)[-k:][::-1]  # Descending order
    top_k_distances = flat_distances[top_k_flat_indices]

    # Convert flat indices back to original indices
    if len(distances.shape) == 2:
        # Convert flat indices to (batch, seq_len) coordinates
        _, seq_len = distances.shape
        top_k_indices = []
        for flat_idx in top_k_flat_indices:
            batch_idx = flat_idx // seq_len
            seq_idx = flat_idx % seq_len
            top_k_indices.append((batch_idx, seq_idx))
    else:
        top_k_indices = top_k_flat_indices.tolist()

    return top_k_indices, top_k_distances.tolist()


def compare_layer_outputs(
    orig_outputs: dict[str, torch.Tensor], comp_outputs: dict[str, torch.Tensor], k: int, metric: str = "l2"
) -> list[LayerComparison]:
    """Compare outputs between original and compressed models."""
    comparisons = []

    for layer_name in orig_outputs:
        if layer_name not in comp_outputs:
            print(f"Warning: Layer {layer_name} not found in compressed model outputs")
            continue

        orig_out = orig_outputs[layer_name]
        comp_out = comp_outputs[layer_name]

        # Calculate all distances
        distances = calculate_distance(orig_out, comp_out, metric)

        # Find top-k differences
        top_k_indices, top_k_distances = find_top_k_differences(orig_out, comp_out, k, metric)

        # Calculate statistics
        mean_distance = float(np.mean(distances))
        max_distance = float(np.max(distances))

        comparison = LayerComparison(
            layer_name=layer_name,
            top_k_indices=top_k_indices,
            top_k_distances=top_k_distances,
            mean_distance=mean_distance,
            max_distance=max_distance,
            distance_metric=metric,
        )
        comparisons.append(comparison)

    return comparisons


def prepare_sample_inputs(tokenizer, num_samples: int = 5, max_length: int = 64):
    """Prepare sample inputs for inference."""
    # Load sample data
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test", revision="b08601e")
    dataset = dataset.filter(lambda x: len(x["text"]) > max_length)

    inputs = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        text = example["text"][: max_length * 4]  # Take more text to ensure we have enough tokens
        tokenized = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        inputs.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "text": text[:100] + "..." if len(text) > 100 else text,
            }
        )

    return inputs


def print_results(comparisons: list[LayerComparison], tokenizer, sample_inputs: list[dict]):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 80)
    print("LAYER COMPARISON RESULTS")
    print("=" * 80)

    for comp in comparisons:
        print(f"\nLayer: {comp.layer_name}")
        print(f"Distance metric: {comp.distance_metric}")
        print(f"Mean distance: {comp.mean_distance:.6f}")
        print(f"Max distance: {comp.max_distance:.6f}")
        print(f"\nTop-{len(comp.top_k_indices)} rows with largest differences:")

        for i, (idx, distance) in enumerate(zip(comp.top_k_indices, comp.top_k_distances)):
            if isinstance(idx, tuple):
                batch_idx, seq_idx = idx
                print(f"  {i + 1}. Batch {batch_idx}, Position {seq_idx}: distance = {distance:.6f}")

                # Try to show the corresponding token
                if batch_idx < len(sample_inputs):
                    input_ids = sample_inputs[batch_idx]["input_ids"]
                    if seq_idx < input_ids.shape[1]:
                        token_id = input_ids[0, seq_idx].item()
                        token = tokenizer.decode([token_id])
                        print(f"      Token: '{token}' (ID: {token_id})")
            else:
                print(f"  {i + 1}. Row {idx}: distance = {distance:.6f}")
        print("-" * 60)


def convert_compression_params(params_dict: dict) -> dict:
    """Convert string mode parameters to proper NNCF enum values."""
    # Create a copy to avoid modifying the original
    converted_params = params_dict.copy()

    # Convert mode string to enum if present
    if "mode" in converted_params:
        mode_str = converted_params["mode"]
        if isinstance(mode_str, str):
            # Map string values to enum values
            mode_mapping = {
                "int8_sym": CompressWeightsMode.INT8_SYM,
                "int8_asym": CompressWeightsMode.INT8_ASYM,
                "int4_sym": CompressWeightsMode.INT4_SYM,
                "int4_asym": CompressWeightsMode.INT4_ASYM,
                "nf4": CompressWeightsMode.NF4,
                "e2m1": CompressWeightsMode.E2M1,
            }

            if mode_str.lower() in mode_mapping:
                converted_params["mode"] = mode_mapping[mode_str.lower()]
            else:
                available_modes = ", ".join(mode_mapping.keys())
                msg = f"Unknown compression mode: {mode_str}. Available modes: {available_modes}"
                raise ValueError(msg)

    return converted_params


def main():
    parser = argparse.ArgumentParser(description="Compare outputs between original and compressed LLM models")
    parser.add_argument(
        "--model-id",
        type=str,
        default="TinyLlama/TinyLlama_v1.1",
        help="Hugging Face model ID or path to OpenVINO model",
    )
    parser.add_argument(
        "--backend", type=str, default="pytorch", choices=["pytorch", "openvino"], help="Model backend to use"
    )
    parser.add_argument(
        "--layers", type=str, nargs="+", default=["embed", "lm_head"], help="Layer name patterns to analyze"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top different rows to show")
    parser.add_argument(
        "--metric", type=str, default="l2", choices=["l2", "cosine", "mse"], help="Distance metric to use"
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of input samples to test")
    parser.add_argument("--max-length", type=int, default=32, help="Maximum sequence length")
    parser.add_argument(
        "--compression-params",
        type=str,
        default='{"mode": "int4_sym", "group_size": 128}',
        help="JSON string with compression parameters",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")

    args = parser.parse_args()

    # Convert backend string to enum
    backend = BackendType.PYTORCH if args.backend == "pytorch" else BackendType.OPENVINO

    print(f"Loading model: {args.model_id}")
    print(f"Backend: {backend.value}")

    # Load tokenizer (always from HuggingFace for text processing)
    if backend == BackendType.OPENVINO and not args.model_id.startswith("./") and not args.model_id.startswith("/"):
        # For OpenVINO, we still need the tokenizer from HuggingFace
        tokenizer_id = args.model_id
    else:
        # For local OpenVINO models, try to infer tokenizer or use a default
        tokenizer_id = args.model_id if backend == BackendType.PYTORCH else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load original model
    print("Loading original model...")
    original_model = load_model(args.model_id, backend, args.device)

    # Find target layers
    target_layers = find_layer_names(original_model, args.layers, backend)
    if not target_layers:
        return

    print(f"Found layers to analyze: {target_layers}")

    # Create compressed model
    print("Creating compressed model...")
    import json

    compression_params_raw = json.loads(args.compression_params)
    compression_params = convert_compression_params(compression_params_raw)

    # Prepare calibration dataset for compression
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", revision="b08601e")
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)

    def transform_fn(data):
        tokenized = tokenizer(data["text"], return_tensors="pt", max_length=128, truncation=True)
        # For OpenVINO, convert to numpy
        if backend == BackendType.OPENVINO:
            return {"input_ids": tokenized["input_ids"].numpy()}
        return {"input_ids": tokenized["input_ids"]}

    calibration_dataset = nncf.Dataset(dataset.select(range(100)), transform_fn)

    # Load model again for compression (instead of copying)
    print("Loading model for compression...")
    model_for_compression = load_model(args.model_id, backend, args.device)
    compressed_model = compress_model(model_for_compression, backend, compression_params, calibration_dataset)

    # Prepare sample inputs
    print(f"Preparing {args.num_samples} sample inputs...")
    sample_inputs = prepare_sample_inputs(tokenizer, args.num_samples, args.max_length)

    # Set up output extractors
    orig_extractor = ModelOutputExtractor(original_model, target_layers, backend, calibration_dataset)
    comp_extractor = ModelOutputExtractor(compressed_model, target_layers, backend, calibration_dataset)

    # For OpenVINO, try to collect activations using NNCF statistics
    if backend == BackendType.OPENVINO:
        print("Attempting to collect activations using NNCF statistics...")
        orig_activations = orig_extractor.collect_openvino_activations()
        comp_activations = comp_extractor.collect_openvino_activations()

        if orig_activations or comp_activations:
            print(
                f"Successfully collected activations for {len(orig_activations)} original "
                f"and {len(comp_activations)} compressed operations"
            )
        else:
            print("No activations collected via NNCF statistics - falling back to basic inference")

    all_comparisons = []

    try:
        # Process each sample
        for i, sample in enumerate(sample_inputs):
            print(f"\nProcessing sample {i + 1}/{len(sample_inputs)}")
            print(f"Text preview: {sample['text']}")

            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]

            if backend == BackendType.PYTORCH:
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)

            # Get outputs from both models
            _, orig_outputs = orig_extractor(input_ids=input_ids, attention_mask=attention_mask)
            _, comp_outputs = comp_extractor(input_ids=input_ids, attention_mask=attention_mask)

            # Compare outputs
            comparisons = compare_layer_outputs(orig_outputs, comp_outputs, args.top_k, args.metric)

            print(f"Sample {i + 1} - Layer comparison summary:")
            for comp in comparisons:
                print(f"  {comp.layer_name}: mean_dist={comp.mean_distance:.6f}, max_dist={comp.max_distance:.6f}")

            all_comparisons.extend(comparisons)

        # Print detailed results
        if all_comparisons:
            print_results(all_comparisons, tokenizer, sample_inputs)

    finally:
        # Clean up hooks
        orig_extractor.remove_hooks()
        comp_extractor.remove_hooks()

    print(f"\nAnalysis complete! Processed {len(sample_inputs)} samples across {len(target_layers)} layers.")


if __name__ == "__main__":
    main()
