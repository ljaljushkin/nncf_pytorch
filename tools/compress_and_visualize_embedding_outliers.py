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
Script to compress TinyLlama model using nncf.compress_weights and visualize
outliers in token embedding weights before and after compression.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf import CompressWeightsMode
from nncf.parameters import StripFormat

# Optional seaborn for better styling
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def extract_embedding_weights(model, tokenizer, layer_name: str = "embed_tokens") -> tuple[np.ndarray, list[str]]:
    """
    Extract token embedding weights from the model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        layer_name: Name of the embedding layer

    Returns:
        Tuple of (weights array, token list)
    """
    # Find the embedding layer
    embedding_layer = None
    for name, module in model.named_modules():
        if layer_name in name and hasattr(module, "weight"):
            embedding_layer = module
            print(f"Found embedding layer: {name}")
            break

    if embedding_layer is None:
        msg = f"Could not find embedding layer with name containing '{layer_name}'"
        raise ValueError(msg)

    # Get weights and convert to numpy
    weights = embedding_layer.weight.detach().cpu().numpy()
    print(f"Embedding weights shape: {weights.shape}")

    # Get token list (limited to actual vocabulary)
    vocab_size = min(weights.shape[0], tokenizer.vocab_size)
    tokens = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            tokens.append(token)
        except Exception:
            tokens.append(f"<unk_{i}>")

    return weights[:vocab_size], tokens


def calculate_outlier_metrics(weights: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Calculate outlier scores for embedding weights.

    Args:
        weights: Embedding weights matrix [vocab_size, hidden_dim]
        method: Method to calculate outliers ('zscore', 'magnitude', 'variance')

    Returns:
        Array of outlier scores for each token
    """
    if method == "zscore":
        # Z-score based on L2 norm of each token embedding
        norms = np.linalg.norm(weights, axis=1)
        z_scores = np.abs((norms - np.mean(norms)) / np.std(norms))
        return z_scores

    elif method == "magnitude":
        # Simple L2 norm magnitude
        return np.linalg.norm(weights, axis=1)

    elif method == "variance":
        # Variance across dimensions for each token
        return np.var(weights, axis=1)

    elif method == "max_abs":
        # Maximum absolute value across dimensions
        return np.max(np.abs(weights), axis=1)

    else:
        msg = f"Unknown outlier method: {method}"
        raise ValueError(msg)


def find_top_outliers(outlier_scores: np.ndarray, tokens: list[str], top_k: int = 20) -> list[tuple[str, float, int]]:
    """
    Find top-k outlier tokens.

    Args:
        outlier_scores: Array of outlier scores
        tokens: List of token strings
        top_k: Number of top outliers to return

    Returns:
        List of (token, score, index) tuples
    """
    indices = np.argsort(outlier_scores)[-top_k:][::-1]  # Descending order
    return [(tokens[i], outlier_scores[i], i) for i in indices]


def create_outlier_visualization(
    original_scores: np.ndarray,
    compressed_scores: np.ndarray,
    tokens: list[str],
    top_k: int = 20,
    method: str = "zscore",
    output_dir: str = "output",
) -> None:
    """
    Create comprehensive visualization of embedding outliers.

    Args:
        original_scores: Outlier scores for original model
        compressed_scores: Outlier scores for compressed model
        tokens: List of token strings
        top_k: Number of top outliers to visualize
        method: Outlier detection method used
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    if HAS_SEABORN:
        sns.set_palette("husl")

    # 1. Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Token Embedding Outlier Analysis ({method.upper()})", fontsize=16, fontweight="bold")

    # Score distributions
    axes[0, 0].hist(original_scores, bins=50, alpha=0.7, label="Original", density=True)
    axes[0, 0].hist(compressed_scores, bins=50, alpha=0.7, label="Compressed", density=True)
    axes[0, 0].set_xlabel("Outlier Score")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Outlier Score Distributions")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Score change (compressed - original)
    score_diff = compressed_scores - original_scores
    axes[0, 1].hist(score_diff, bins=50, alpha=0.7, color="red")
    axes[0, 1].set_xlabel("Score Change (Compressed - Original)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Change in Outlier Scores")
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter plot: original vs compressed scores
    axes[1, 0].scatter(original_scores, compressed_scores, alpha=0.6, s=1)
    axes[1, 0].plot(
        [original_scores.min(), original_scores.max()], [original_scores.min(), original_scores.max()], "r--", alpha=0.8
    )
    axes[1, 0].set_xlabel("Original Outlier Score")
    axes[1, 0].set_ylabel("Compressed Outlier Score")
    axes[1, 0].set_title("Original vs Compressed Scores")
    axes[1, 0].grid(True, alpha=0.3)

    # Top outliers comparison
    top_original = find_top_outliers(original_scores, tokens, top_k)
    top_compressed = find_top_outliers(compressed_scores, tokens, top_k)

    # Create comparison of top outliers
    top_indices = list(set([idx for _, _, idx in top_original] + [idx for _, _, idx in top_compressed]))[:top_k]
    comparison_data = []

    for idx in top_indices:
        if idx < len(tokens):
            # Clean token display for plotting
            token = tokens[idx]
            # Replace problematic characters for display
            display_token = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            display_token = display_token[:15]  # Truncate long tokens
            comparison_data.append(
                {
                    "Token": display_token,
                    "Original": original_scores[idx],
                    "Compressed": compressed_scores[idx],
                    "Index": idx,
                }
            )

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Original", ascending=False).head(top_k)

        x_pos = np.arange(len(df))
        width = 0.35

        axes[1, 1].bar(x_pos - width / 2, df["Original"], width, label="Original", alpha=0.8)
        axes[1, 1].bar(x_pos + width / 2, df["Compressed"], width, label="Compressed", alpha=0.8)
        axes[1, 1].set_xlabel("Token")
        axes[1, 1].set_ylabel("Outlier Score")
        axes[1, 1].set_title(f"Top {top_k} Outliers Comparison")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(df["Token"], rotation=45, ha="right")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/embedding_outliers_analysis_{method}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Detailed heatmap of top outliers
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Get embeddings for top outliers for heatmap
    top_outlier_indices = [idx for _, _, idx in top_original[: min(20, len(top_original))]]
    if len(top_outlier_indices) > 0:
        # This would need the actual embedding weights to create meaningful heatmaps
        print("Top outlier tokens (Original model):")
        for i, (token, score, idx) in enumerate(top_original[:20]):
            display_token = repr(token) if any(c in token for c in ["\n", "\r", "\t"]) else token
            print(f"{i + 1:2d}. {display_token:20s} (idx: {idx:5d}, score: {score:.4f})")

        print("\nTop outlier tokens (Compressed model):")
        for i, (token, score, idx) in enumerate(top_compressed[:20]):
            display_token = repr(token) if any(c in token for c in ["\n", "\r", "\t"]) else token
            print(f"{i + 1:2d}. {display_token:20s} (idx: {idx:5d}, score: {score:.4f})")

    plt.close()  # Close the heatmap figure for now

    # 3. Statistical summary
    print(f"\n=== Outlier Analysis Summary ({method.upper()}) ===")
    print(f"Total tokens analyzed: {len(original_scores)}")
    print(f"Original model - Mean outlier score: {np.mean(original_scores):.4f}")
    print(f"Original model - Std outlier score: {np.std(original_scores):.4f}")
    print(f"Compressed model - Mean outlier score: {np.mean(compressed_scores):.4f}")
    print(f"Compressed model - Std outlier score: {np.std(compressed_scores):.4f}")
    print(f"Mean score change: {np.mean(score_diff):.4f}")
    print(f"Std score change: {np.std(score_diff):.4f}")

    # Calculate correlation
    correlation = np.corrcoef(original_scores, compressed_scores)[0, 1]
    print(f"Correlation between original and compressed scores: {correlation:.4f}")


def prepare_calibration_dataset(tokenizer, dataset_size: int = 128, max_length: int = 512):
    """
    Prepare calibration dataset for compression.

    Args:
        tokenizer: The tokenizer
        dataset_size: Number of samples to use
        max_length: Maximum sequence length

    Returns:
        NNCF Dataset object
    """
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", revision="b08601e")
    dataset = dataset.filter(lambda example: len(example["text"]) > max_length // 2)

    def transform_fn(data):
        tokenized = tokenizer(data["text"], return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        return {"input_ids": tokenized["input_ids"]}

    return nncf.Dataset(dataset.select(range(dataset_size)), transform_fn)


def compress_model_weights(
    model,
    tokenizer,
    compression_mode: CompressWeightsMode = CompressWeightsMode.INT4_SYM,
    ratio: float = 0.8,
    group_size: int = 128,
    dataset_size: int = 128,
):
    """
    Compress model weights using NNCF.

    Args:
        model: PyTorch model to compress
        tokenizer: Associated tokenizer
        compression_mode: Compression mode to use
        ratio: Compression ratio
        group_size: Group size for quantization
        dataset_size: Size of calibration dataset

    Returns:
        Compressed model
    """
    print("Preparing calibration dataset...")
    calibration_dataset = prepare_calibration_dataset(tokenizer, dataset_size)

    # print(f"Compressing model with mode={compression_mode}, ratio={ratio}, group_size={group_size}")
    print(f"Compressing model with mode={compression_mode}")

    # Apply compression
    compressed_model = nncf.compress_weights(
        model,
        dataset=calibration_dataset,
        mode=compression_mode,
        # ratio=ratio,
        # group_size=group_size,
        # We want to keep embeddings in analysis, so don't ignore them completely
        # ignored_scope=nncf.IgnoredScope(patterns=[".*lm_head.*"]),  # Only ignore output head
        ignored_scope=nncf.IgnoredScope(patterns=["(?!.*embed.*)\w+"]),
        # all_layers=True,  # Include embedding layers in compression
    )

    return compressed_model, calibration_dataset


def main():
    parser = argparse.ArgumentParser(description="Compress TinyLlama and visualize embedding outliers")
    parser.add_argument(
        "-m", "--model-id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face model ID"
    )
    parser.add_argument(
        "--compression-mode",
        type=str,
        default="int8_asym",
        choices=["int4_sym", "int4_asym", "int8_sym", "int8_asym", "nf4"],
        help="Compression mode",
    )
    parser.add_argument("--ratio", type=float, default=0.8, help="Compression ratio (0.0 to 1.0)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument(
        "--outlier-method",
        type=str,
        default="zscore",
        choices=["zscore", "magnitude", "variance", "max_abs"],
        help="Method to calculate outlier scores",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of top outliers to analyze")
    parser.add_argument("--dataset-size", type=int, default=128, help="Size of calibration dataset")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="embedding_outlier_analysis",
        help="Output directory for plots and results",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")

    args = parser.parse_args()

    # Convert compression mode string to enum
    mode_mapping = {
        "int4_sym": CompressWeightsMode.INT4_SYM,
        "int4_asym": CompressWeightsMode.INT4_ASYM,
        "int8_sym": CompressWeightsMode.INT8_SYM,
        "int8_asym": CompressWeightsMode.INT8_ASYM,
        "nf4": CompressWeightsMode.NF4,
    }
    compression_mode = mode_mapping[args.compression_mode]

    print(f"Loading model: {args.model_id}")
    print(f"Device: {args.device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load original model
    original_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, device_map=args.device
    )
    original_model.eval()

    print("Extracting original embedding weights...")
    original_weights, tokens = extract_embedding_weights(original_model, tokenizer)
    original_outlier_scores = calculate_outlier_metrics(original_weights, args.outlier_method)

    print("Compressing model...")
    compressed_model, calibration_dataset = compress_model_weights(
        original_model,
        tokenizer,
        compression_mode=compression_mode,
        ratio=args.ratio,
        group_size=args.group_size,
        dataset_size=args.dataset_size,
    )
    compressed_model = nncf.strip(
        compressed_model,
        do_copy=False,
        strip_format=StripFormat.IN_PLACE,
        example_input=original_model.dummy_inputs,  # calibration_dataset.get_inference_data([0]),
    )

    print("Extracting compressed embedding weights...")
    compressed_weights, _ = extract_embedding_weights(compressed_model, tokenizer)
    compressed_outlier_scores = calculate_outlier_metrics(compressed_weights, args.outlier_method)

    print("Creating visualizations...")
    create_outlier_visualization(
        original_outlier_scores,
        compressed_outlier_scores,
        tokens,
        top_k=args.top_k,
        method=args.outlier_method,
        output_dir=args.output_dir,
    )

    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)

    # Create detailed analysis CSV
    results_data = []
    for i, token in enumerate(tokens):
        if i < len(original_outlier_scores) and i < len(compressed_outlier_scores):
            # Clean token for CSV compatibility
            clean_token = repr(token) if "\n" in token or "\r" in token or '"' in token else token
            results_data.append(
                {
                    "token_idx": i,
                    "token": clean_token,
                    "original_score": original_outlier_scores[i],
                    "compressed_score": compressed_outlier_scores[i],
                    "score_change": compressed_outlier_scores[i] - original_outlier_scores[i],
                    "score_ratio": compressed_outlier_scores[i] / (original_outlier_scores[i] + 1e-8),
                }
            )

    df = pd.DataFrame(results_data)
    df.to_csv(f"{args.output_dir}/outlier_analysis_results.csv", index=False, escapechar="\\")  # Save top outliers
    top_original = find_top_outliers(original_outlier_scores, tokens, args.top_k)
    top_compressed = find_top_outliers(compressed_outlier_scores, tokens, args.top_k)

    with open(f"{args.output_dir}/top_outliers_summary.txt", "w") as f:
        f.write("Embedding Outlier Analysis Results\n")
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Compression: {args.compression_mode}, ratio={args.ratio}, group_size={args.group_size}\n")
        f.write(f"Outlier method: {args.outlier_method}\n\n")

        f.write("TOP OUTLIERS (Original Model):\n")
        for i, (token, score, idx) in enumerate(top_original):
            f.write(f"{i + 1:2d}. {token:25s} (idx: {idx:5d}, score: {score:.6f})\n")

        f.write("\nTOP OUTLIERS (Compressed Model):\n")
        for i, (token, score, idx) in enumerate(top_compressed):
            f.write(f"{i + 1:2d}. {token:25s} (idx: {idx:5d}, score: {score:.6f})\n")

    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")
    print(f"- Visualization: embedding_outliers_analysis_{args.outlier_method}.png")
    print("- Detailed data: outlier_analysis_results.csv")
    print("- Summary: top_outliers_summary.txt")


if __name__ == "__main__":
    main()
