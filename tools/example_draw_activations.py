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
Example script demonstrating how to use the draw_activations_2 function
for comparing floating point and quantized model activations with histograms.
"""

import os
import re
from typing import Any, Optional

import numpy as np
import openvino as ov
from datasets import load_dataset
from transformers import AutoTokenizer

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.statistics.collectors import get_raw_stat_collector


def gen_pkv(n_vals, sz, f_size=12):  # opt-125m
    res = {}
    # "?,12,?,64"
    for i in range(n_vals):
        res[f"past_key_values.{i}.value"] = np.zeros((1, f_size, 0, sz))
        res[f"past_key_values.{i}.key"] = np.zeros((1, f_size, 0, sz))

    return res


def gen_pkv_(n_vals, sz):  # redpajama
    res = {}
    # "?,32,?,128"
    for i in range(n_vals):
        res[f"past_key_values.{i}.value"] = np.zeros((1, 32, 0, sz))
        res[f"past_key_values.{i}.key"] = np.zeros((1, 32, 0, sz))

    return res


def transform_fn(data, model, tokenizer):
    tokenized_text = tokenizer(data["text"], return_tensors="np")
    input_ids = tokenized_text["input_ids"]
    attention_mask = tokenized_text["attention_mask"]

    batch_size = input_ids.shape[0]
    input_shapes = {}
    for val in model.model.inputs:
        name = val.any_name
        shape = list(val.partial_shape.get_min_shape())
        shape[0] = batch_size
        input_shapes[name] = shape

    inputs = {}
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = tokenized_text["attention_mask"]
    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1

    # The magic forms KV cache as model inputs
    batch_size = input_ids.shape[0]
    for input_name in model.key_value_input_names:
        model_inputs = model.model.input(input_name)
        shape = model_inputs.get_partial_shape()
        shape[0] = batch_size
        if shape[2].is_dynamic:
            shape[2] = 0
        else:
            shape[1] = 0
        inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())

    if "beam_idx" in input_shapes:
        inputs["beam_idx"] = np.arange(batch_size, dtype=int)

    inputs["position_ids"] = position_ids
    return inputs


def get_noop_statistic_collector(
    num_samples: int, channel_axis: int, window_size: Optional[int] = None, inplace: bool = True
):
    """
    Raw statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :param inplace: Whether the mean reducer should be calculated inplace or out of place.
    :return: Mean statistic collector.
    """
    from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
    from nncf.experimental.common.tensor_statistics.collectors import RawReducer
    from nncf.experimental.common.tensor_statistics.collectors import TensorCollector

    reducer = RawReducer()
    aggregate_mean = NoopAggregator(num_samples)

    collector = TensorCollector()
    collector.register_statistic_branch("MEAN_STAT", reducer, aggregate_mean)
    return collector


def get_statistic_points(model, graph, nodes, subset_size) -> StatisticPointsContainer:
    statistic_container = StatisticPointsContainer()
    OUTPUT_PORT_OF_NODE = 0
    INPUT_PORT_OF_NODE = 0

    # Collection of statistics after/before layers.
    for node in nodes:
        node_name = node.node_name
        channel_axis = node.metatype.output_channel_axis
        if channel_axis is None:
            channel_axis = -1

        # For layers with weights, there is only one output port - 0.
        statistic_point_out = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_name, port_id=OUTPUT_PORT_OF_NODE)
        stat_collector_out = get_raw_stat_collector(num_samples=subset_size)
        statistic_container.add_statistic_point(
            StatisticPoint(target_point=statistic_point_out, tensor_collector=stat_collector_out, algorithm="collect")
        )

        # For layers with weights, there is only one output port - 0.
        statistic_point_in = OVTargetPoint(TargetType.PRE_LAYER_OPERATION, node_name, port_id=INPUT_PORT_OF_NODE)
        stat_collector_in = get_raw_stat_collector(num_samples=subset_size)
        statistic_container.add_statistic_point(
            StatisticPoint(target_point=statistic_point_in, tensor_collector=stat_collector_in, algorithm="collect")
        )

    return statistic_container


def get_input_statistic_points(model, graph, nodes, subset_size) -> StatisticPointsContainer:
    statistic_container = StatisticPointsContainer()
    INPUT_PORT_OF_NODE = 1
    OUTPUT_PORT_OF_NODE = 0

    # Collection of statistics after/before layers.
    for node in nodes:
        node_name = node.node_name
        channel_axis = node.metatype.output_channel_axis
        if channel_axis is None:
            channel_axis = -1

        # For layers with weights, there is only one output port - 0.
        statistic_point_in = OVTargetPoint(TargetType.PRE_LAYER_OPERATION, node_name, port_id=INPUT_PORT_OF_NODE)
        stat_collector_in = get_noop_statistic_collector(
            channel_axis=channel_axis, num_samples=subset_size, inplace=False
        )
        statistic_container.add_statistic_point(
            StatisticPoint(target_point=statistic_point_in, tensor_collector=stat_collector_in, algorithm="collect")
        )

        # For layers with weights, there is only one output port - 0.
        statistic_point_out = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_name, port_id=OUTPUT_PORT_OF_NODE)
        stat_collector_out = get_noop_statistic_collector(
            channel_axis=channel_axis, num_samples=subset_size, inplace=False
        )
        statistic_container.add_statistic_point(
            StatisticPoint(target_point=statistic_point_out, tensor_collector=stat_collector_out, algorithm="collect")
        )

    return statistic_container


def collect_activations(model: "ov.Model", dataset: Any, regexp: re.Pattern, subset_size=4):
    """
    Collect activations from OpenVINO model using NNCF statistics collector.

    Args:
        model: OpenVINO model
        dataset: NNCF dataset for calibration
        regexp: Regular expression pattern to match layer names
        subset_size: Number of samples to process

    Returns:
        Dictionary with layer outputs for matched operations
    """

    # Find target operations
    target_ops = [op for op in model.get_ordered_ops() if bool(regexp.search(op.get_friendly_name()))]

    if len(target_ops) == 0:
        print("No operations found matching pattern:", regexp.pattern)
        return {}

    print(f"Found {len(target_ops)} target operations matching pattern")

    # Create NNCF graph
    graph = GraphConverter.create_nncf_graph(model)
    statistics_aggregator = OVStatisticsAggregator(dataset)

    node_keys = graph.get_all_node_keys()
    target_names = [val for val in node_keys if bool(regexp.search(val))]
    target_ops = [graph.get_node_by_key(target_name) for target_name in target_names]

    statistic_points = get_statistic_points(model, graph, target_ops, subset_size)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, graph)

    res = {}
    for k, v in statistics_aggregator.statistic_points.items():
        res[k] = {
            "in": list(v[1].algorithm_to_tensor_collectors["collect"][0].aggregators.values())[0]._container,
            "out": list(v[0].algorithm_to_tensor_collectors["collect"][0].aggregators.values())[0]._container,
        }

    return res


def collect_weights(model: "ov.Model", dataset: Any, regexp: "re.Pattern", subset_size=1):
    """
    Collect weights from OpenVINO model using NNCF statistics collector.

    Args:
        model: OpenVINO model
        dataset: NNCF dataset for calibration
        regexp: Regular expression pattern to match layer names
        subset_size: Number of samples to process

    Returns:
        Dictionary with layer weights for matched operations
    """
    # Find target operations
    target_ops = [op for op in model.get_ordered_ops() if bool(regexp.search(op.get_friendly_name()))]

    if len(target_ops) == 0:
        print("No operations found matching pattern:", regexp.pattern)
        return {}

    graph = GraphConverter.create_nncf_graph(model)
    statistics_aggregator = OVStatisticsAggregator(dataset)

    node_keys = graph.get_all_node_keys()
    target_names = [val for val in node_keys if bool(regexp.search(val))]
    target_ops = [graph.get_node_by_key(target_name) for target_name in target_names]

    statistic_points = get_input_statistic_points(model, graph, target_ops, subset_size)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, graph)

    res = {}
    for k, v in statistics_aggregator.statistic_points.items():
        res[k] = {
            "in": list(v[0].algorithm_to_tensor_collectors["collect"][0].aggregators.values())[0]._container,
            "out": list(v[1].algorithm_to_tensor_collectors["collect"][0].aggregators.values())[0]._container,
        }

    return res


def draw_activations_2(model_name_fp, model_name_int, tokenizer_name, n_samples=1):
    """
    Draw activation histograms comparing floating point and integer models.

    Args:
        model_name_fp: Path to floating point model
        model_name_int: Path to integer model
        tokenizer_name: Path to tokenizer
        n_samples: Number of samples to process
    """
    from functools import partial

    import matplotlib.pyplot as plt
    from optimum.intel.openvino import OVModelForCausalLM

    from nncf import Dataset

    # def transform_fn(data, model, tokenizer):
    #     """Transform function for NNCF dataset."""
    #     tokenized = tokenizer(data["text"], return_tensors="pt", max_length=128, truncation=True)
    #     return {"input_ids": tokenized["input_ids"].numpy()}

    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    dataset = dataset.filter(lambda example: len(example["text"]) > 60)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model_fp = OVModelForCausalLM.from_pretrained(model_name_fp, trust_remote_code=True)
    model_int = OVModelForCausalLM.from_pretrained(model_name_int, trust_remote_code=True)
    nncf_dataset = Dataset(dataset, partial(transform_fn, model=model_fp, tokenizer=tokenizer))

    reg_names = [
        # r'__module.transformer.h.\d+.mlp.fc_in/aten::linear/MatMul_',
        # r"__module.model.layers..*\d+.self_attn.q_proj/ov_ext::linear/MatMul",
        # r"__module.model.layers.0.self_attn.q_proj/aten::linear/MatMul",
        # r"__module.model.layers..*\d+.mlp.down_proj/ov_ext::linear/MatMul",
        # r'__module.transformer.h.\d+.attn.out_proj/aten::linear/MatMul_',
        # r'__module.transformer.h.\d+.mlp.fc_out/aten::linear/MatMul_'
        # r"__module.model.layers.\d+.self_attn/aten::unsqueeze/Unsqueeze_2"
        # r"__module.model.layers.0.mlp.gate_proj/ov_ext::linear/MatMul"
        # r"__module.model.layers.\d+.self_attn.q_proj/aten::linear/MatMul"
        # r"__module.model.embed_tokens/aten::embedding/Gather",
        # r"__module.lm_head/aten::linear/MatMul",
        r"__module.model.layers.\d+.mlp.down_proj/aten::linear/MatMul"
        # r"__module.model.layers.\d+.self_attn.q_proj/aten::linear/MatMul",
    ]

    for reg_name in reg_names:
        regexp = re.compile(reg_name)
        activations_fp = collect_activations(model_fp.model, nncf_dataset, regexp, n_samples)
        activations_int = collect_activations(model_int.model, nncf_dataset, regexp, n_samples)

        act_types = ["in", "out"]

        for act_type in act_types:
            print("REG Name", reg_name, act_type)

            plot_data = {
                "int": {"mean": [], "std": []},
                "fp": {"mean": [], "std": []},
                "fp-int": {"mean": [], "std": [], "relative": [], "relative_comp": []},
            }

            for k in activations_fp:
                int_data = [np.array(arr.data) for arr in activations_int[k][act_type]]
                fp_data = [np.array(arr.data) for arr in activations_fp[k][act_type]]
                try:
                    int_data = np.concatenate(int_data, axis=1)
                    fp_data = np.concatenate(fp_data, axis=1)
                except Exception:
                    try:
                        int_data = np.concatenate(int_data, axis=2)
                        fp_data = np.concatenate(fp_data, axis=2)
                    except Exception:
                        int_data = np.concatenate(int_data, axis=3)
                        fp_data = np.concatenate(fp_data, axis=3)

                fig, (ax1) = plt.subplots(1)
                fig.suptitle(k)

                ax1.hist(int_data.flatten(), bins=100, alpha=0.5, label=f"{k}_fp8")
                ax1.hist(fp_data.flatten(), bins=100, alpha=0.5, label=f"{k}_fp16")
                print(k)
                print("compressed ", int_data.flatten().min(), int_data.flatten().max())
                print("fp32 ", fp_data.flatten().min(), fp_data.flatten().max())
                plt.legend(loc="best")
                plt.show()

                plot_data["int"]["mean"].append(np.mean(int_data))
                plot_data["fp"]["mean"].append(np.mean(fp_data))
                plot_data["fp-int"]["mean"].append(np.mean(fp_data - int_data))
                plot_data["fp-int"]["relative"].append(
                    np.mean(np.abs(fp_data - int_data) / (np.abs(fp_data) + 0.00000001))
                )

                plot_data["int"]["std"].append(np.std(int_data))
                plot_data["fp"]["std"].append(np.std(fp_data))
                plot_data["fp-int"]["std"].append(np.std(fp_data - int_data))

            x = [i for i in range(len(activations_fp))]

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            fig.suptitle("Distributions")

            ax1.errorbar(x, plot_data["fp"]["mean"], plot_data["fp"]["std"], linestyle="none", marker="^")
            ax1.set_title("fp")

            ax2.errorbar(
                x,
                plot_data["int"]["mean"],
                plot_data["int"]["std"],
                linestyle="none",
                marker="^",
                fmt="o",
                ecolor="green",
            )
            ax2.set_title("int")

            ax3.errorbar(
                x,
                plot_data["fp-int"]["mean"],
                plot_data["fp-int"]["std"],
                linestyle="none",
                marker="^",
                fmt="o",
                ecolor="red",
            )
            ax3.set_title("fp-int")

            ax4.plot(x, plot_data["fp-int"]["relative"], color="green", label="int")
            ax4.set_title("fp-int relative")

            plt.legend()
            # Save the plot to a file
            output_dir = "activation_plots"
            os.makedirs(output_dir, exist_ok=True)
            safe_name = (
                reg_name.replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_").replace("*", "_")
            )
            plt.savefig(f"{output_dir}/{safe_name}_{act_type}.png")
            plt.close()


# def main():
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="Compare activations between floating point and quantized OpenVINO models with histograms"
#     )
#     parser.add_argument(
#         "--model-fp",
#         type=str,
#         help="Path to floating point OpenVINO model directory",
#         default="/home/nlyaly/projects/nncf/tests/post_training/tmp/fp32_models/TinyLlama__TinyLlama_v1.1",
#     )
#     parser.add_argument(
#         "--model-int",
#         type=str,
#         help="Path to quantized integer OpenVINO model directory",
#         default="/home/nlyaly/projects/nncf/tests/post_training/tmp/tinyllama_int8_data_free/TORCH",
#     )
#     parser.add_argument(
#         "--tokenizer",
#         type=str,
#         help="HuggingFace tokenizer name or path",
#         default="TinyLlama/TinyLlama-1.1B",
#     )
#     parser.add_argument("--n-samples", type=int, default=10, help="Number of samples to process (default: 1)")

#     args = parser.parse_args()

#     print("🚀 Starting activation comparison...")
#     print(f"Floating point model: {args.model_fp}")
#     print(f"Integer model: {args.model_int}")
#     print(f"Tokenizer: {args.tokenizer}")
#     print(f"Number of samples: {args.n_samples}")
#     print()

#     try:
#         # Call the draw_activations_2 function
#         draw_activations_2(
#             model_name_fp=args.model_fp,
#             model_name_int=args.model_int,
#             tokenizer_name=args.tokenizer,
#             n_samples=args.n_samples,
#         )

#         print("✅ Analysis completed successfully!")
#         print("📊 Histogram plots should have been displayed showing:")
#         print("   - Individual activation distributions for each matched layer")
#         print("   - Comparison plots with error bars")
#         print("   - Relative difference plots")

#         return 0

#     except Exception as e:
#         print(f"❌ Error during analysis: {e}")
#         import traceback

#         traceback.print_exc()
#         return 1


# if __name__ == "__main__":
#     exit_code = main()
#     exit(exit_code)

model_fp = "/home/nlyaly/projects/nncf/tests/post_training/tmp/fp32_models/TinyLlama__TinyLlama_v1.1"
model_int = "/home/nlyaly/projects/nncf/tests/post_training/tmp/tinyllama_int8_data_free/TORCH"
model_int_ign_emb = "/home/nlyaly/projects/nncf/tests/post_training/tmp/tinyllama_int8_data_free_ign_emb/TORCH"
tokenizer = "TinyLlama/TinyLlama_v1.1"
# model_fp = "/home/nlyaly/projects/nncf/tests/post_training/tmp/fp32_models/meta-llama__Llama-3.2-1B"
# model_int = "/home/nlyaly/projects/nncf/tests/post_training/tmp/tinyllama_int4_data_free/TORCH"
# tokenizer = "meta-llama/Llama-3.2-1B"

# model_fp = "/home/nlyaly/projects/nncf/tests/post_training/tmp/fp32_models/google__gemma-2-2b"
# model_int = (
#     "/home/nlyaly/projects/nncf/tests/post_training/tmp/tinyllama_data_aware_gptq_scale_estimation_stateful/TORCH"
# )
# tokenizer = "google/gemma-2-2b"
model_fp = "/home/nlyaly/projects/nncf/tests/post_training/tmp/fp32_models/TinyLlama__TinyLlama-1.1B-Chat-v1.0"
# model_int = "/home/nlyaly/projects/nncf/tests/post_training/tmp/tinyllama_int8_data_free/TORCH"
tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
n_samples = 2
draw_activations_2(
    model_name_fp=model_fp,
    model_name_int=model_int,
    tokenizer_name=tokenizer,
    n_samples=n_samples,
)
