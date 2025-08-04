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
Demonstration script showing how to use the compress_and_visualize_embedding_outliers.py tool.
This script runs the embedding outlier analysis with TinyLlama model.
"""

import os
import subprocess
import sys


def run_embedding_analysis():
    """Run the embedding outlier analysis with default parameters."""

    # Check if the main script exists
    script_path = os.path.join(os.path.dirname(__file__), "compress_and_visualize_embedding_outliers.py")
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False

    # Basic command with default parameters
    cmd = [
        sys.executable,
        script_path,
        "--model-id",
        "TinyLlama/TinyLlama_v1.1",
        "--compression-mode",
        "int8_asym",
        # "--ratio",
        # "0.8",
        # "--group-size",
        # "128",
        "--outlier-method",
        "variance",
        # "zscore",
        # "max_abs",
        # "magnitude",
        "--top-k",
        "20",
        "--dataset-size",
        "1",  # Smaller dataset for faster demo
        "--output-dir",
        "demo_embedding_analysis",
        "--device",
        "cpu",
    ]

    print("Running embedding outlier analysis with TinyLlama...")
    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("1. Load TinyLlama model")
    print("2. Extract original embedding weights")
    print("3. Compress the model using NNCF")
    print("4. Extract compressed embedding weights")
    print("5. Analyze outliers and create visualizations")
    print("\nStarting analysis...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Analysis completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed with error code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def run_different_methods():
    """Run the analysis with different outlier detection methods."""

    script_path = os.path.join(os.path.dirname(__file__), "compress_and_visualize_embedding_outliers.py")
    methods = ["zscore", "magnitude", "variance", "max_abs"]

    print("\nRunning analysis with different outlier detection methods...")

    for method in methods:
        print(f"\n--- Running with {method} method ---")
        cmd = [
            sys.executable,
            script_path,
            "--model-id",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--compression-mode",
            "int4_sym",
            "--ratio",
            "0.8",
            "--outlier-method",
            method,
            "--top-k",
            "10",
            "--dataset-size",
            "32",  # Very small for quick demo
            "--output-dir",
            f"demo_analysis_{method}",
            "--device",
            "cpu",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ {method} method completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ {method} method failed with error code {e.returncode}")
            print("Error:", e.stderr[:200], "..." if len(e.stderr) > 200 else "")


def main():
    print("🔬 TinyLlama Embedding Outlier Analysis Demo")
    print("=" * 50)

    # choice = input(
    #     "\nChoose analysis type:\n"
    #     "1. Basic analysis (default settings)\n"
    #     "2. Multiple methods comparison\n"
    #     "Enter choice (1 or 2): "
    # ).strip()

    # if choice == "2":
    # run_different_methods()
    # else:
    run_embedding_analysis()

    print("\n" + "=" * 50)
    print("Demo completed! Check the output directories for results.")


if __name__ == "__main__":
    main()
