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
Test script for model output comparison tool.
Simple example showing how to use the compare_model_outputs.py tool.
"""

import subprocess
import sys
from pathlib import Path


def run_comparison_test():
    """Run a simple test of the model comparison tool."""

    tool_path = Path(__file__).parent / "compare_model_outputs.py"

    # Basic test with TinyLlama model
    cmd = [
        sys.executable,
        str(tool_path),
        "--model-id",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--layers",
        "embed_tokens",
        "lm_head",
        "--top-k",
        "5",
        "--metric",
        "l2",
        "--num-samples",
        "2",
        "--max-length",
        "16",
        "--compression-params",
        '{"mode": "int4_sym", "group_size": 128}',
        "--device",
        "cpu",
    ]

    print("Running model comparison test...")
    print("Command:", " ".join(cmd))
    print("-" * 80)

    try:
        subprocess.run(cmd, capture_output=False, text=True, check=True)
        print("\nTest completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTest failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = run_comparison_test()
    sys.exit(0 if success else 1)
