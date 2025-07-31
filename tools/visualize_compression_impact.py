#!/usr/bin/env python3
"""
Copyright (c) 2024 Intel Corporation
Licensed under         if hasattr(module, 'weight') and 'embed' in name.lower() and module.weight.dim() == 2:e Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Script to visualize the tokens most affected by NNCF compression.
Focuses on tokens with the largest embedding changes and degradation patterns.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoModel
from transformers import AutoTokenizer

from nncf.parameters import StripFormat

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add NNCF to path if needed
current_dir = Path(__file__).parent
nncf_dir = current_dir.parent / "nncf"
if nncf_dir.exists():
    sys.path.insert(0, str(nncf_dir))

try:
    import nncf
    from nncf import CompressWeightsMode
    from nncf import compress_weights
except ImportError as e:
    print(f"❌ Error importing NNCF: {e}")
    print("Please install NNCF: pip install nncf")
    sys.exit(1)


def load_model_and_tokenizer(model_name: str) -> tuple[torch.nn.Module, Any]:
    """Load model and tokenizer."""
    print(f"📥 Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ Model loaded successfully")
        print(f"   - Vocabulary size: {len(tokenizer)}")
        print(f"   - Embedding dimension: {model.config.hidden_size}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def extract_embedding_weights(model: torch.nn.Module) -> torch.Tensor:
    """Extract embedding weights from the model."""
    for name, module in model.named_modules():
        if (
            hasattr(module, "weight") and "embed" in name.lower() and module.weight.dim() == 2
        ):  # Standard embedding layer
            print(f"📊 Found embedding layer: {name}")
            print(f"   - Shape: {module.weight.shape}")
            return module.weight.data.clone()

    msg = "No embedding layer found in the model"
    raise ValueError(msg)


def prepare_calibration_data(tokenizer: Any, dataset_size: int = 128) -> list[str]:
    """Prepare calibration dataset for compression."""
    print(f"📚 Preparing calibration data ({dataset_size} samples)...")

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        # Filter out empty texts and take the first dataset_size samples
        texts = []
        for item in dataset:
            text = item["text"].strip()
            if text and len(text) > 50:  # Filter out very short texts
                texts.append(text)
                if len(texts) >= dataset_size:
                    break

        print(f"✅ Prepared {len(texts)} calibration samples")
        return texts

    except Exception as e:
        print(f"❌ Error loading calibration data: {e}")
        raise


def compress_model_weights(
    model: torch.nn.Module, tokenizer: Any, compression_mode: str, dataset_size: int = 128
) -> torch.nn.Module:
    """Compress model weights using NNCF."""
    print(f"🔄 Compressing model with {compression_mode}...")

    # Map string mode to NNCF enum
    mode_map = {
        "int4_sym": CompressWeightsMode.INT4_SYM,
        "int4_asym": CompressWeightsMode.INT4_ASYM,
        "int8_sym": CompressWeightsMode.INT8_SYM,
        "int8_asym": CompressWeightsMode.INT8_ASYM,
        "nf4": CompressWeightsMode.NF4,
    }

    if compression_mode not in mode_map:
        msg = f"Unsupported compression mode: {compression_mode}"
        raise ValueError(msg)

    compression_mode_enum = mode_map[compression_mode]

    # Prepare calibration data
    calibration_texts = prepare_calibration_data(tokenizer, dataset_size)

    def transform_fn(data):
        """Transform function for calibration data."""
        tokenized = tokenizer(data["text"], return_tensors="pt", max_length=512, truncation=True, padding=True)
        return {"input_ids": tokenized["input_ids"]}

    # Create NNCF dataset
    from datasets import Dataset

    hf_dataset = Dataset.from_dict({"text": calibration_texts})
    calibration_dataset = nncf.Dataset(hf_dataset, transform_fn)

    # Compress the model
    try:
        compressed_model = compress_weights(
            model,
            mode=compression_mode_enum,
            dataset=calibration_dataset,
            ignored_scope=nncf.IgnoredScope(patterns=["(?!.*embed.*)\w+"]),
        )
        compressed_model = nncf.strip(
            compressed_model,
            do_copy=False,
            strip_format=StripFormat.IN_PLACE,
            example_input=compressed_model.dummy_inputs,  # calibration_dataset.get_inference_data([0]),
        )
        print(f"✅ Model compressed successfully with {compression_mode}")
        return compressed_model

    except Exception as e:
        print(f"❌ Error during compression: {e}")
        raise


def calculate_compression_impact(
    original_weights: torch.Tensor, compressed_weights: torch.Tensor, tokenizer: Any
) -> pd.DataFrame:
    """Calculate detailed compression impact metrics for each token."""
    print("📊 Calculating compression impact metrics...")

    # Ensure weights are on CPU for calculations
    orig_w = original_weights.cpu().numpy()
    comp_w = compressed_weights.cpu().numpy()

    vocab_size = orig_w.shape[0]
    impact_data = []

    for token_id in range(vocab_size):
        orig_vec = orig_w[token_id]
        comp_vec = comp_w[token_id]

        # Calculate various impact metrics
        l2_distance = np.linalg.norm(orig_vec - comp_vec)
        cosine_distance = 1 - np.dot(orig_vec, comp_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(comp_vec) + 1e-8)
        magnitude_change = abs(np.linalg.norm(comp_vec) - np.linalg.norm(orig_vec))
        mse = np.mean((orig_vec - comp_vec) ** 2)
        variance_change = abs(np.var(comp_vec) - np.var(orig_vec))
        max_abs_diff = np.max(np.abs(orig_vec - comp_vec))

        # Get token string (handle special tokens safely)
        try:
            token_str = tokenizer.decode([token_id])
            # Clean token for display (remove special characters that might cause issues)
            token_display = repr(token_str) if any(ord(c) > 127 for c in token_str) else token_str
        except Exception:
            token_display = f"<UNK_{token_id}>"

        impact_data.append(
            {
                "token_id": token_id,
                "token": token_display,
                "l2_distance": l2_distance,
                "cosine_distance": cosine_distance,
                "magnitude_change": magnitude_change,
                "mse": mse,
                "variance_change": variance_change,
                "max_abs_diff": max_abs_diff,
                "orig_norm": np.linalg.norm(orig_vec),
                "comp_norm": np.linalg.norm(comp_vec),
            }
        )

    df = pd.DataFrame(impact_data)

    # Calculate overall impact score (weighted combination of metrics)
    df["impact_score"] = (
        0.3 * (df["l2_distance"] / df["l2_distance"].max())
        + 0.2 * (df["cosine_distance"] / df["cosine_distance"].max())
        + 0.2 * (df["magnitude_change"] / df["magnitude_change"].max())
        + 0.2 * (df["mse"] / df["mse"].max())
        + 0.1 * (df["variance_change"] / df["variance_change"].max())
    )

    print(f"✅ Calculated impact metrics for {len(df)} tokens")
    return df


def find_most_affected_tokens(impact_df: pd.DataFrame, top_k: int = 50) -> dict[str, pd.DataFrame]:
    """Find tokens most affected by compression across different metrics."""
    print(f"🔍 Finding top {top_k} most affected tokens...")

    metrics = ["impact_score", "l2_distance", "cosine_distance", "magnitude_change", "mse"]
    top_tokens = {}

    for metric in metrics:
        top_tokens[metric] = impact_df.nlargest(top_k, metric)[
            ["token", "token_id", metric, "orig_norm", "comp_norm"]
        ].copy()

    return top_tokens


def create_comprehensive_visualization(
    impact_df: pd.DataFrame, top_tokens: dict[str, pd.DataFrame], compression_mode: str, output_dir: str
):
    """Create comprehensive visualization of compression impact."""
    print("📈 Creating compression impact visualizations...")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle(f"Compression Impact Analysis - {compression_mode.upper()}", fontsize=16, fontweight="bold", y=0.98)

    # 1. Impact Score Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    impact_df["impact_score"].hist(bins=50, alpha=0.7, color="skyblue", ax=ax1)
    mean_val = impact_df["impact_score"].mean()
    std_val = impact_df["impact_score"].std()
    ax1.axvline(mean_val, color="red", linestyle="-", alpha=0.8, label=f"Mean: {mean_val:.4f}")
    ax1.axvline(mean_val + std_val, color="orange", linestyle="--", alpha=0.6, label=f"+1σ: {mean_val + std_val:.4f}")
    ax1.set_title("Impact Score Distribution")
    ax1.set_xlabel("Impact Score")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. L2 Distance vs Cosine Distance
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(
        impact_df["l2_distance"],
        impact_df["cosine_distance"],
        c=impact_df["impact_score"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    ax2.set_title("L2 Distance vs Cosine Distance")
    ax2.set_xlabel("L2 Distance")
    ax2.set_ylabel("Cosine Distance")
    plt.colorbar(scatter, ax=ax2, label="Impact Score")
    ax2.grid(True, alpha=0.3)

    # 3. Magnitude Change Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    impact_df["magnitude_change"].hist(bins=50, alpha=0.7, color="lightcoral", ax=ax3)
    ax3.set_title("Magnitude Change Distribution")
    ax3.set_xlabel("Magnitude Change")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    # 4. Top tokens by impact score (bar plot)
    ax4 = fig.add_subplot(gs[1, :])
    top_impact = top_tokens["impact_score"].head(20)
    bars = ax4.barh(range(len(top_impact)), top_impact["impact_score"], color="steelblue")
    ax4.set_yticks(range(len(top_impact)))
    ax4.set_yticklabels(
        [
            f"{row['token'][:15]}..." if len(str(row["token"])) > 15 else str(row["token"])
            for _, row in top_impact.iterrows()
        ],
        fontsize=8,
    )
    ax4.set_title("Top 20 Tokens by Impact Score")
    ax4.set_xlabel("Impact Score")
    ax4.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(
            width * 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            weight="bold",
        )

    # 5. Correlation heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    metrics_for_corr = ["l2_distance", "cosine_distance", "magnitude_change", "mse", "variance_change"]
    corr_matrix = impact_df[metrics_for_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, ax=ax5, cbar_kws={"shrink": 0.8})
    ax5.set_title("Metric Correlations")

    # 6. MSE vs L2 Distance
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(impact_df["mse"], impact_df["l2_distance"], alpha=0.6, s=10, color="purple")
    ax6.set_title("MSE vs L2 Distance")
    ax6.set_xlabel("MSE")
    ax6.set_ylabel("L2 Distance")
    ax6.grid(True, alpha=0.3)

    # 7. Norm ratio analysis
    ax7 = fig.add_subplot(gs[2, 2])
    norm_ratio = impact_df["comp_norm"] / (impact_df["orig_norm"] + 1e-8)
    norm_ratio.hist(bins=50, alpha=0.7, color="gold", ax=ax7)
    ax7.axvline(1.0, color="red", linestyle="-", alpha=0.8, label="No change")
    ax7.set_title("Norm Ratio Distribution\n(Compressed/Original)")
    ax7.set_xlabel("Norm Ratio")
    ax7.set_ylabel("Frequency")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis("off")

    stats_text = f"""
    📊 COMPRESSION IMPACT SUMMARY

    Overall Statistics:
    • Total tokens analyzed: {len(impact_df):,}
    • Mean impact score: {impact_df["impact_score"].mean():.4f} ± {impact_df["impact_score"].std():.4f}
    • Tokens with high impact (>95th percentile): {len(impact_df[impact_df["impact_score"] > impact_df["impact_score"].quantile(0.95)]):,}

    Distance Metrics:
    • Mean L2 distance: {impact_df["l2_distance"].mean():.4f}
    • Mean cosine distance: {impact_df["cosine_distance"].mean():.4f}
    • Mean magnitude change: {impact_df["magnitude_change"].mean():.4f}

    Most Affected Token: "{top_tokens["impact_score"].iloc[0]["token"]}" (Score: {top_tokens["impact_score"].iloc[0]["impact_score"]:.4f})
    """

    ax8.text(
        0.05,
        0.95,
        stats_text,
        transform=ax8.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Save the overview plot
    overview_path = os.path.join(output_dir, "compression_impact_overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # Create detailed plot for top affected tokens
    create_detailed_token_plot(top_tokens, compression_mode, output_dir)

    print(f"✅ Visualizations saved to {output_dir}/")


def create_detailed_token_plot(top_tokens: dict[str, pd.DataFrame], compression_mode: str, output_dir: str):
    """Create detailed visualization focusing on top affected tokens."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Top Affected Tokens by Different Metrics - {compression_mode.upper()}", fontsize=14, fontweight="bold"
    )

    metrics = ["impact_score", "l2_distance", "cosine_distance", "magnitude_change", "mse"]
    colors = ["steelblue", "darkgreen", "darkred", "purple", "orange"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i // 3, i % 3]

        top_data = top_tokens[metric].head(15)
        bars = ax.barh(range(len(top_data)), top_data[metric], color=color, alpha=0.7)

        ax.set_yticks(range(len(top_data)))
        ax.set_yticklabels(
            [
                f"{row['token'][:20]}..." if len(str(row["token"])) > 20 else str(row["token"])
                for _, row in top_data.iterrows()
            ],
            fontsize=8,
        )
        ax.set_title(f"Top 15 by {metric.replace('_', ' ').title()}")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width * 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold",
            )

    # Hide the last subplot if we have fewer than 6 metrics
    if len(metrics) < 6:
        axes[1, 2].axis("off")

    plt.tight_layout()

    detail_path = os.path.join(output_dir, "top_affected_tokens_detail.png")
    plt.savefig(detail_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def save_detailed_results(
    impact_df: pd.DataFrame, top_tokens: dict[str, pd.DataFrame], compression_mode: str, output_dir: str
):
    """Save detailed results to CSV and text files."""
    print("💾 Saving detailed results...")

    # Save full impact data to CSV
    csv_path = os.path.join(output_dir, "compression_impact_detailed.csv")
    impact_df.to_csv(csv_path, index=False, escapechar="\\")

    # Save summary to text file
    summary_path = os.path.join(output_dir, "compression_impact_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"COMPRESSION IMPACT ANALYSIS SUMMARY\n")
        f.write(f"Compression Mode: {compression_mode.upper()}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"Total tokens analyzed: {len(impact_df):,}\n")
        f.write(f"Mean impact score: {impact_df['impact_score'].mean():.6f}\n")
        f.write(f"Std impact score: {impact_df['impact_score'].std():.6f}\n")
        f.write(f"Max impact score: {impact_df['impact_score'].max():.6f}\n")
        f.write(f"95th percentile: {impact_df['impact_score'].quantile(0.95):.6f}\n")
        f.write(
            f"Highly affected tokens (>95th percentile): {len(impact_df[impact_df['impact_score'] > impact_df['impact_score'].quantile(0.95)]):,}\n\n"
        )

        f.write("DISTANCE METRICS:\n")
        for metric in ["l2_distance", "cosine_distance", "magnitude_change", "mse", "variance_change"]:
            f.write(
                f"{metric}: mean={impact_df[metric].mean():.6f}, std={impact_df[metric].std():.6f}, max={impact_df[metric].max():.6f}\n"
            )
        f.write("\n")

        f.write("TOP 20 MOST AFFECTED TOKENS BY IMPACT SCORE:\n")
        f.write("-" * 60 + "\n")
        for i, (_, row) in enumerate(top_tokens["impact_score"].head(20).iterrows(), 1):
            f.write(f"{i:2d}. Token: {row['token']:<20} | Score: {row['impact_score']:.6f} | ID: {row['token_id']}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("For detailed per-token analysis, see: compression_impact_detailed.csv\n")

    print(f"✅ Results saved:")
    print(f"   - Detailed CSV: {csv_path}")
    print(f"   - Summary text: {summary_path}")


def main():
    """Main function to run compression impact analysis."""
    parser = argparse.ArgumentParser(description="Visualize tokens most affected by NNCF compression")
    parser.add_argument(
        "-m", "--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name or path"
    )
    parser.add_argument(
        "--compression_mode",
        type=str,
        default="int8_asym",
        choices=["int4_sym", "int4_asym", "int8_sym", "int8_asym", "nf4"],
        help="Compression mode",
    )
    parser.add_argument("--dataset_size", type=int, default=128, help="Size of calibration dataset")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top affected tokens to analyze")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="compression_demo_chat", help="Output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting compression impact analysis...")
    print(f"   Model: {args.model}")
    print(f"   Compression: {args.compression_mode}")
    print(f"   Dataset size: {args.dataset_size}")
    print(f"   Output: {args.output_dir}")
    print()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model)

        # Extract original embedding weights
        original_weights = extract_embedding_weights(model)

        # Compress the model
        compressed_model = compress_model_weights(model, tokenizer, args.compression_mode, args.dataset_size)

        # Extract compressed embedding weights
        compressed_weights = extract_embedding_weights(compressed_model)

        # Calculate impact metrics
        impact_df = calculate_compression_impact(original_weights, compressed_weights, tokenizer)

        # Find most affected tokens
        top_tokens = find_most_affected_tokens(impact_df, args.top_k)

        # Create visualizations
        create_comprehensive_visualization(impact_df, top_tokens, args.compression_mode, args.output_dir)

        # Save detailed results
        save_detailed_results(impact_df, top_tokens, args.compression_mode, args.output_dir)

        print("✅ Analysis complete!")
        print(f"📁 Results saved to: {args.output_dir}/")
        print("   - compression_impact_overview.png")
        print("   - top_affected_tokens_detail.png")
        print("   - compression_impact_detailed.csv")
        print("   - compression_impact_summary.txt")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
