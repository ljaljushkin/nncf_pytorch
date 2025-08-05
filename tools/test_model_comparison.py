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
    # """Run a simple test of the model comparison tool."""

    tool_path = Path(__file__).parent / "compare_model_outputs.py"

    # Test with PyTorch backend
    print("Testing PyTorch backend...")
    cmd_pytorch = [
        sys.executable,
        str(tool_path),
        "--model-id",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--backend",
        "pytorch",
        "--layers",
        "lm_head",
        "model.embed_tokens",
        # "model.layers.0.mlp.gate_proj",
        # "model.layers.0.self_attn.q_proj",
        "--top-k",
        "3",
        "--metric",
        "l2",
        "--num-samples",
        "1",
        "--max-length",
        "128",
        "--compression-params",
        # '{"mode": "int8_sym", "group_size": 128}',
        '{"mode": "int8_asym"}',
        "--device",
        "cpu",
    ]

    print("PyTorch Command:", " ".join(cmd_pytorch))
    print("-" * 80)

    try:
        subprocess.run(cmd_pytorch, capture_output=False, text=True, check=True)
        print("\nPyTorch test completed successfully!")
        pytorch_success = True
    except subprocess.CalledProcessError as e:
        print(f"\nPyTorch test failed with return code {e.returncode}")
        pytorch_success = False
    except Exception as e:
        print(f"\nPyTorch test failed with exception: {e}")
        pytorch_success = False

    # Test with OpenVINO backend (if available)
    # print("\n" + "=" * 80)
    # print("Testing OpenVINO backend...")

    # try:
    #     import openvino

    #     openvino_available = True
    # except ImportError:
    #     print("OpenVINO not available, skipping OpenVINO test")
    #     openvino_available = False
    #     openvino_success = True  # Don't count as failure if not available

    # if openvino_available:
    #     cmd_openvino = [
    #         sys.executable,
    #         str(tool_path),
    #         "--model-id",
    #         "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #         "--backend",
    #         "openvino",
    #         "--layers",
    #         "__module.model.layers.0.mlp.gate_proj/ov_ext::linear/MatMul",
    #         # "up_proj",  # OpenVINO layer names are different
    #         "--top-k",
    #         "3",
    #         "--metric",
    #         "l2",
    #         "--num-samples",
    #         "1",
    #         "--max-length",
    #         "8",
    #         "--compression-params",
    #         '{"mode": "int4_sym", "group_size": 128}',
    #         "--device",
    #         "cpu",
    #     ]

    #     print("OpenVINO Command:", " ".join(cmd_openvino))
    #     print("-" * 80)

    #     try:
    #         subprocess.run(cmd_openvino, capture_output=False, text=True, check=True)
    #         print("\nOpenVINO test completed successfully!")
    #         openvino_success = True
    #     except subprocess.CalledProcessError as e:
    #         print(f"\nOpenVINO test failed with return code {e.returncode}")
    #         openvino_success = False
    #     except Exception as e:
    #         print(f"\nOpenVINO test failed with exception: {e}")
    #         openvino_success = False
    # else:
    #     openvino_success = True

    # return True  # pytorch_success and openvino_success


if __name__ == "__main__":
    success = run_comparison_test()
    sys.exit(0 if success else 1)
