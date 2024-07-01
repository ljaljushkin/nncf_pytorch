# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

# import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import openvino as ov
import pandas as pd
import wandb
from openvino.runtime import opset13 as opset

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.utils import get_reduction_axes
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor.functions import numeric as fns
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters

# from nncf.quantization.algorithms.weight_compression.weight_lowering import _get_nf4_error
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization

# from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    def __init__(self, model: ov.Model, name_to_node_mapping: Dict = None):
        if name_to_node_mapping is None:
            self.name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        else:
            self.name_to_node_mapping = name_to_node_mapping

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVMatMulMetatype]

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.OVConvolutionMetatype,
            om.OVDepthwiseConvolutionMetatype,
            om.OVConvolutionBackpropDataMetatype,
            om.OVGroupConvolutionMetatype,
            om.OVGroupConvolutionBackpropDataMetatype,
        ]

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        channel_axes = get_weight_channel_axes(node_with_weight)
        const_shape = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]
        return get_reduction_axes(channel_axes, const_shape)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        result = []
        for weight_port_id in node.layer_attributes.get_const_port_ids():
            weight_name = node.layer_attributes.constant_attributes[weight_port_id]["name"]
            result.append((weight_name, weight_port_id))
        return result

    def get_weight(
        self,
        node_with_weight: NNCFNode,
        weight_port_id: int,
        model: ov.Model,
        graph: NNCFGraph,
    ) -> Tensor:
        weight_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        weight_tensor = get_const_value(weight_node)
        return Tensor(weight_tensor)

    def set_weight(
        self,
        node_with_weight: NNCFNode,
        weight_port_id: int,
        model: ov.Model,
        graph: NNCFGraph,
        weight: Tensor,
    ):
        node_with_const = self.name_to_node_mapping[node_with_weight.node_name]

        const_port = node_with_const.input(weight_port_id)
        const_node = node_with_const.input_value(weight_port_id).get_node()

        new_const_node = ov.runtime.op.Constant(weight.data, shared_memory=True)
        new_const_node.set_friendly_name(const_node.get_friendly_name())
        const_port.replace_source_output(new_const_node.output(0))

        const_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        self.name_to_node_mapping[const_name] = new_const_node

        new_output = new_const_node.output(0)
        for target_input in const_node.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)

        del const_node

    def _get_int_mul(
        self,
        compressed_weight,
        compression_dtype,
        const_node_name,
        wc_params,
        const_dtype,
    ):
        compressed_const = opset.constant(compressed_weight.tensor.data, dtype=compression_dtype, name=const_node_name)
        converted_const = opset.convert(compressed_const, ov.Type.f16)
        if compressed_weight.zero_point is not None:
            zero_point_const = opset.constant(
                compressed_weight.zero_point.data,
                dtype=compression_dtype,
                name=f"{const_node_name}/zero_point",
            )
            converted_zero_point = opset.convert(zero_point_const, ov.Type.f16)
            converted_const = opset.subtract(
                converted_const,
                converted_zero_point,
                name=f"{const_node_name}/zero_point/subtract",
            )

        scale_const = opset.constant(
            compressed_weight.scale.data,
            dtype=ov.Type.f16,
            name=f"{const_node_name}/scale",
        )
        mul = opset.multiply(
            converted_const,
            scale_const,
            name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}",
        )

        mul = opset.convert(
            mul,
            const_dtype,
            name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}/convert",
        )

        return mul

    def insert_lora_residual(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        wc_params: WeightCompressionParameters,
        weight,
        compressed_weight,
        rank=8,
        int8_lora=True,
        L2QER=False,
        w_regulation=False,
        n_iters=3,
        cached_adapter_weights: Dict[str, np.ndarray] = None,
    ):
        import numpy as np
        import numpy.linalg as linalg
        import scipy.linalg as slinalg

        # import scipy.optimize as optimize

        print(wc_params.node_with_weight.node_name)
        layer_name = wc_params.node_with_weight.node_name  # .split("/")[0].split("__module.model.layers.")[1]

        print(compressed_weight.tensor.shape)
        loss = []

        # if layer_name == '23.mlp.down_proj':
        #     wc_params.compression_config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_ASYM)
        #     return {layer_name: []}
        if layer_name + "__Vr" in cached_adapter_weights and layer_name + "__US" in cached_adapter_weights:  # \
            # and layer_name not in ['23.mlp.down_proj']:#, '5.mlp.down_proj', '16.mlp.down_proj', '22.mlp.down_proj']:
            # NOTE: tune specified layer
            Vr = cached_adapter_weights[layer_name + "__Vr"]
            US = cached_adapter_weights[layer_name + "__US"]
            print(f"Taking adapters from cache for {layer_name}")
        else:
            print(
                f"Tune {layer_name} with rank={rank} w_regulation={w_regulation} L2QER={L2QER} n_iters={n_iters} "
                "int8_lora={int8_lora} num_cached={len(cached_adapter_weights)}"
            )

            q_weights = do_dequantization(
                compressed_weight.tensor,
                compressed_weight.scale,
                compressed_weight.zero_point,
                wc_params.reduction_axes[0],
            )
            q_weights_data = q_weights.data
            # q_weights_data = _get_nf4_error(
            #   compressed_weight.tensor.data, compressed_weight.scale.data, wc_params.reduction_axes[0],
            #   wc_params.compression_config.group_size
            # )

            X = wc_params.X.data
            diff_before = np.mean(np.abs(weight.data @ X - q_weights_data @ X))

            # q_w + USV = w => USV = w - q_w
            residual = (weight.data - q_weights_data).astype(np.float32)
            w_residual = residual.copy()
            if wc_params.reduction_axes == 0:
                residual = np.transpose(residual)
            if wc_params.stat is not None:  # and False:
                s = wc_params.stat.data
                if wc_params.compression_config.group_size > 0:
                    if not L2QER:
                        gs = wc_params.compression_config.group_size
                        n_gs = s.shape[0] // gs
                        for i in range(n_gs):
                            offset = i * gs
                            denum = np.sum(s[offset : offset + gs])
                            s[offset : offset + gs] = s[offset : offset + gs] / denum
                            denum = np.max(s[offset : offset + gs])
                            s[offset : offset + gs] = s[offset : offset + gs] / denum
                        s = np.expand_dims(s, 0)
                        residual = residual * s
                    else:
                        s = s / math.sqrt(fns.min(s) * fns.max(s))
                        print("Scaled S stats: min={:.2f} max={:.2f}".format(fns.min(s), fns.max(s)))
                        s = np.diagflat(s)
                        s_inv = np.linalg.inv(s)
                        print(f"Shapes before mul: residual={residual.shape} s={s.shape}")
                        residual = residual @ s

                # low_k = max(int(2 * s.shape[0] // 3), 1)
                # lowk_idxs = np.argsort(s.data)[:low_k]
                # for idx in lowk_idxs:
                #     residual[:, idx] = 0.0

            svd = linalg.svd(residual, compute_uv=True, full_matrices=False)
            U = svd[0]
            S = svd[1]
            V = svd[2]

            Ur = U[:, :rank]
            Sr = np.diag(S[:rank])
            Vr = V[:rank, :]

            # NOTE: our init
            if not L2QER:
                Vr = Sr @ Vr
                US = Ur
            else:
                print(f"Shapes: residual={residual.shape} s={s.shape} s_inv={s_inv.shape} Vr={Vr.shape}")
                print(f"Shapes: Ur={Ur.shape} X={X.shape}")
                US = Ur @ Sr
                Vr = Vr @ s_inv
                print(f"Shapes: US={US.shape}")

            if wc_params.X is not None:  # rectification by data
                X = wc_params.X.data
                dY = w_residual @ X  # [O, C] * [C, 1] = [O, 1]
                # NOTE: noise for regularization - DOESN'T WORK!!!
                # dY =  dY + 0.001 * np.random.randn(*dY.shape)
                # US @ Vr = res
                # US @ Vr @ X = dY
                # US @ |VR VR @ X| = |res dY|

                """
                1 Rectification 1:  0.010965054096178758 0.0007162072519819473 0.0006860411346467133
                1 Rectification 2:  0.010965054096178758 0.0007162072519819473 0.0006725860367021748
                2 Rectification 1:  0.010965054096178758 0.0006725860367021748 0.0006652183148589173
                2 Rectification 2:  0.010965054096178758 0.0006725860367021748 0.0006605420416146037
                Before:  0.0001609714  After:  0.00017714252 8
                """
                for i in range(n_iters):
                    VX = Vr @ X
                    if not w_regulation:
                        sol = slinalg.lstsq(np.transpose(VX), np.transpose(dY))
                    else:
                        # VrVX = np.concatenate((Vr, VX), axis=1)
                        # # dYR = np.concatenate((w_residual, dY), axis=1)
                        # # NOTE: L2QER
                        # dYR = np.concatenate((residual, dY), axis=1)
                        # sol = slinalg.lstsq(np.transpose(VrVX), np.transpose(dYR))
                        VrVX = np.concatenate((Vr, VX), axis=1)
                        dYR = np.concatenate((w_residual, dY), axis=1)
                        sol = slinalg.lstsq(np.transpose(VrVX), np.transpose(dYR), lapack_driver="gelsy")

                    diff_after_svd = np.mean(np.abs(weight.data @ X - q_weights_data @ X - (US @ Vr) @ X))
                    if i == 0:
                        loss.extend([diff_before, diff_after_svd])
                        wandb.log({layer_name: diff_before})
                        wandb.log({layer_name: diff_after_svd})

                    US = np.transpose(sol[0])

                    diff_after_svd_rectification = np.mean(np.abs(weight.data @ X - q_weights_data @ X - (US @ Vr) @ X))
                    loss.append(diff_after_svd_rectification)
                    wandb.log({layer_name: diff_after_svd_rectification})
                    # if n_iters - i < 3:
                    print(
                        f"{i} Rectification 1: ",
                        diff_before,
                        diff_after_svd,
                        diff_after_svd_rectification,
                    )

                    USI = linalg.pinv(US)
                    if not w_regulation:
                        dYU = USI @ dY
                        sol = slinalg.lstsq(np.transpose(X), np.transpose(dYU), lapack_driver="gelsy")
                    else:
                        # # I = np.ones_like(Vr)
                        # I = np.eye(Vr.shape[1]) # Vr = [r, C], I = [C, C], dY=[O, B]
                        # IX = np.concatenate((I, X), axis=1) # I=[C, C], X=[C, B], IX=[C, C+B]
                        # # dYR = [r,C] + [r, B] = [r, C+B]
                        # dYR = np.concatenate((USI @ residual, USI @ dY), axis=1) # USI=[r, O], residual=[O, C]
                        # sol = slinalg.lstsq(np.transpose(IX), np.transpose(dYR), lapack_driver='gelsy')
                        Ind = np.eye(Vr.shape[1])
                        IX = np.concatenate((Ind, X), axis=1)
                        dYR = np.concatenate((USI @ w_residual, USI @ dY), axis=1)
                        sol = slinalg.lstsq(np.transpose(IX), np.transpose(dYR), lapack_driver="gelsy")

                    Vr = np.transpose(sol[0])

                    diff_after_svd_rectification = np.mean(np.abs(weight.data @ X - q_weights_data @ X - (US @ Vr) @ X))
                    loss.append(diff_after_svd_rectification)
                    wandb.log({layer_name: diff_after_svd_rectification})
                    # if n_iters - i < 3:
                    print(
                        f"{i} Rectification 2: ",
                        diff_before,
                        diff_after_svd,
                        diff_after_svd_rectification,
                    )
            new_residual = US @ Vr
            print(
                "Before: ",
                np.mean(np.abs(residual)),
                " After: ",
                np.mean(np.abs(residual - new_residual)),
                rank,
            )
            weight_delta = np.mean(np.abs(residual)) - np.mean(np.abs(residual - new_residual))
            original_l1_noise = diff_before
            l1_noise_delta = diff_before - diff_after_svd_rectification
            wandb.log(
                {
                    "weight_delta": weight_delta,
                    "original_l1_noise": original_l1_noise,
                    "l1_noise_delta": l1_noise_delta,
                }
            )

            cached_adapter_weights[layer_name + "__Vr"] = Vr
            cached_adapter_weights[layer_name + "__US"] = US

        V = Vr
        input_node = self.name_to_node_mapping[wc_params.node_with_weight.node_name].input_value(0)
        mm_node = self.name_to_node_mapping[wc_params.node_with_weight.node_name]

        const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
        const_node_name = const_attributes["name"]
        const_node = self.name_to_node_mapping[const_node_name]
        const_dtype = const_node.output(0).get_element_type()

        if int8_lora:
            compression_config = WeightCompressionConfig()
            V = Tensor(V)
            compressed_V = compress_weight(
                V,
                wc_params.reduction_axes,
                compression_config,
            )

            US = Tensor(US)
            compressed_US = compress_weight(
                US,
                wc_params.reduction_axes,
                compression_config,
            )
            V_W = self._get_int_mul(
                compressed_V,
                ov.Type.u8,
                wc_params.node_with_weight.node_name + "_V",
                wc_params,
                const_dtype,
            )

            V_MM = opset.matmul(input_node, V_W, transpose_a=False, transpose_b=True)

            US_W = self._get_int_mul(
                compressed_US,
                ov.Type.u8,
                wc_params.node_with_weight.node_name + "_U",
                wc_params,
                const_dtype,
            )

            US_MM = opset.matmul(V_MM, US_W, transpose_a=False, transpose_b=True)
        else:
            V_W = opset.constant(V)
            V_MM = opset.matmul(input_node, V_W, transpose_a=False, transpose_b=True)

            US_W = opset.constant(US)
            US_MM = opset.matmul(V_MM, US_W, transpose_a=False, transpose_b=True)

        node_output_port = mm_node.output(0)
        node_output_source_ports = node_output_port.get_target_inputs()

        add = opset.add(mm_node, US_MM)

        for node_output_source_port in node_output_source_ports:
            node_output_source_port.replace_source_output(add.output(0))

        return {layer_name: loss}

    def transform_model(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        lora=True,
        num_params=None,
    ) -> ov.Model:
        ids = range(num_params)
        data = []
        # model_name = "phi3"
        model_name = "stablelm-1.6b"
        exp_name = "lora_int4_synth"
        # cached_adapter_weights_stablelm-1.6b__lora_nf4_rank8_synth.npz
        CACHE_FILENAME = Path(f"cached_adapter_weights_{model_name}__{exp_name}.npz")
        cached_adapter_weights = dict()
        # if CACHE_FILENAME.exists():
        #     cached_adapter_weights = dict(np.load(CACHE_FILENAME))
        try:
            L2QER = False
            w_regulation = False
            n_iters = 3
            rank = 8
            mixed_rank = False

            wandb_run = wandb.init(
                project="lora_rectify",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=exp_name,
                # Track hyperparameters and run metadata
                config={
                    "model_name": model_name,
                    "total_num_params": num_params,
                    "num_params_to_rectify": len(ids),
                    "rank": rank,
                    "n_iters": n_iters,
                    "w_regulation": w_regulation,
                    "L2QER": L2QER,
                    "mixed_rank": mixed_rank,
                },
            )

            # int4_params = list(filter(lambda x: x.compression_config.num_bits == 4, weight_compression_parameters))

            # NOISE_FILE = Path('max_var_scores.csv')
            # NOISE_FILE = Path(
            #     "/home/nlyaly/projects/lm-evaluation-harness/cache/phi-3-mini-4k-instruct/int4_sym_g128_r100_data_lora
            # 50_int8_anti_maxact/max_var_scores.csv"
            # )
            # if NOISE_FILE.exists():
            #     df = pd.read_csv(NOISE_FILE)
            #     list_diff_before = list(df[df.columns[1]])
            # else:
            #     raise RuntimeError('No noises.csv for criteria!')
            #     import numpy as np
            #     list_diff_before = []
            #     for wc_params in int4_params:
            #         const_attributes =
            #   wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
            #         const_node_name = const_attributes["name"]
            #         const_node = self.name_to_node_mapping[const_node_name]
            #         weight = Tensor(get_const_value(const_node))
            #         shape = weight.shape
            #         q_weights, _, _ = do_integer_quantization(
            # weight, wc_params.reduction_axes[0], wc_params.compression_config)
            #         q_weights = q_weights.reshape(shape)
            #         X = wc_params.X.data
            #         diff_before = np.mean(np.abs(weight.data @ X - q_weights_data @ X))
            #         list_diff_before.append(diff_before)

            #     names = []
            #     for wc_params in int4_params:
            #         pattern = r'layers\.(\d+\.\w+\.\w+)'
            #         match = re.search(pattern, wc_params.node_with_weight.node_name)
            #         if match:
            #             layer_name = match.group(1)
            #         else:
            #             raise RuntimeError('Cant parse: ', wc_params.node_with_weight.node_name)
            #         names.append(layer_name)

            # pd.DataFrame(list_diff_before).to_csv(NOISE_FILE)
            # print(len(names), len(list_diff_before))
            # print(names, sep='\n')
            # print(list_diff_before, sep='\n')
            # TODO: why different sizes???
            #     # pd.DataFrame.from_dict({'name': names, 'orig_noise': list_diff_before}).to_csv(NOISE_FILE)
            #     plt.plot(list_diff_before, title='L1 quantization noise with criteria')
            #     plt.savefig('noises.png')

            # ratio = 0.8
            # criteria = False
            # # n_select = int(num_params * ratio)
            # indexes_of_layers_in_ascending_order_of_scores = [
            #     i[0] for i in sorted(enumerate(list_diff_before), reverse=criteria, key=lambda x: x[1])
            # ]
            # ids = []
            # num_all_weights = sum(wp.num_weights for wp in int4_params)
            # num_weights_in_4bit = 0
            # for index in indexes_of_layers_in_ascending_order_of_scores:
            #     weight_param = int4_params[index]
            #     current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_all_weights
            #     if current_ratio >= ratio:
            #         break
            #     ids.append(index)
            #     num_weights_in_4bit += weight_param.num_weights
            # ids = indexes_of_layers_in_ascending_order_of_scores[:n_select]

            # TODO: work with losses
            # num_per_type_phi = {"qkv_proj": 0, "o_proj": 0, "gate_up_proj": 0, "down_proj": 0}
            # num_per_type_lm = {
            #     "down_proj": 0,
            #     "up_proj": 0,
            #     "gate_proj": 0,
            #     "q_proj": 0,
            #     "k_proj": 0,
            #     "v_proj": 0,
            #     "o_proj": 0,
            # }
            # num_per_type = num_per_type_phi
            # for index, wc_params in enumerate(int4_params):
            #     pattern = r"layers\.(\d+\.\w+\.\w+)"
            #     layer_name = wc_params.node_with_weight.node_name
            #     match = re.search(pattern, layer_name)
            #     if match:
            #         layer_name = match.group(1)
            #     is_selected = index in ids
            #     if is_selected:
            #         for key in num_per_type:
            #             if key in layer_name:
            #                 num_per_type[key] += 1
            #     status = "LORA_8" if is_selected else "LORA_256"
            #     # diff = list_diff_before[index]
            #     # print(f'{layer_name} {status} {diff}')
            #     print(f"{layer_name} {status}")

            # num_per_type = {key.replace("_proj", ""): value for key, value in num_per_type.items()}
            # print("Statistics for LoRA r8 layers: ", num_per_type)

            index = 0
            for wc_params in weight_compression_parameters:
                # lora = True
                # if wc_params.compression_config.num_bits == 4 and
                # wc_params.compression_config.mode == CompressWeightsMode.INT4_ASYM:
                #     wc_params.compression_config.mode = CompressWeightsMode.NF4
                # lora = True

                compression_config = wc_params.compression_config
                if compression_config.mode == CompressWeightsMode.NF4:
                    compression_dtype = ov.Type.nf4
                elif compression_config.mode in [
                    CompressWeightsMode.INT8_ASYM,
                    CompressWeightsMode.INT8_SYM,
                    CompressWeightsMode.INT8,
                    CompressWeightsMode.INT4_ASYM,
                    CompressWeightsMode.INT4_SYM,
                ]:
                    if compression_config.mode in [
                        CompressWeightsMode.INT4_ASYM,
                        CompressWeightsMode.INT4_SYM,
                    ]:
                        compression_dtype = ov.Type.u4
                    else:
                        compression_dtype = ov.Type.u8
                else:
                    raise ValueError(f"{compression_config.mode.value} is not supported.")

                const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[
                    wc_params.weight_port_id
                ]
                const_node_name = const_attributes["name"]
                const_node = self.name_to_node_mapping[const_node_name]
                const_dtype = const_node.output(0).get_element_type()

                weight = Tensor(get_const_value(const_node))
                original_shape = weight.shape
                compressed_weight = compress_weight(
                    weight,
                    wc_params.reduction_axes,
                    compression_config,
                    precomputed_scales[wc_params.node_with_weight.node_name],
                )

                compressed_const = opset.constant(
                    compressed_weight.tensor.data,
                    dtype=compression_dtype,
                    name=const_node_name,
                )
                converted_const = opset.convert(compressed_const, ov.Type.f16)
                if compressed_weight.zero_point is not None:
                    zero_point_const = opset.constant(
                        compressed_weight.zero_point.data,
                        dtype=compression_dtype,
                        name=f"{const_node_name}/zero_point",
                    )
                    converted_zero_point = opset.convert(zero_point_const, ov.Type.f16)
                    converted_const = opset.subtract(
                        converted_const,
                        converted_zero_point,
                        name=f"{const_node_name}/zero_point/subtract",
                    )

                scale_const = opset.constant(
                    compressed_weight.scale.data,
                    dtype=ov.Type.f16,
                    name=f"{const_node_name}/scale",
                )
                mul = opset.multiply(
                    converted_const,
                    scale_const,
                    name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}",
                )

                if compression_config.group_size != -1:
                    mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)

                if const_dtype != ov.Type.f16:
                    mul = opset.convert(
                        mul,
                        const_dtype,
                        name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}/convert",
                    )

                mul_output = mul.output(0)
                for target_input in const_node.output(0).get_target_inputs():
                    target_input.replace_source_output(mul_output)

                if wc_params.compression_config.num_bits == 4 and lora:
                    index += 1
                    rank, w_regulation = (
                        8,
                        False,
                    )  # if index - 1 in ids else (256, True)
                    data.append(
                        self.insert_lora_residual(
                            model,
                            graph,
                            wc_params,
                            weight,
                            compressed_weight,
                            rank=rank,
                            L2QER=L2QER,
                            w_regulation=w_regulation,
                            n_iters=n_iters,
                            cached_adapter_weights=cached_adapter_weights,
                        )
                    )
        finally:
            # pass
            wandb_run.finish()
            df = pd.DataFrame(data)
            df.to_csv("losses.csv")
            # TODO: uncomment to update cache
            np.savez(CACHE_FILENAME, **cached_adapter_weights)

        # reset name_to_node_mapping
        self.name_to_node_mapping = None
        return model

    @staticmethod
    def dump_parameters(
        model: ov.Model,
        parameters: Dict,
        algo_name: Optional[str] = "quantization",
        path: Optional[List] = None,
    ) -> None:
        dump_parameters(model, parameters, algo_name, path)

    @staticmethod
    def get_compress_decompress_pipeline(
        weight_compression_parameter: WeightCompressionParameters,
        w_shape,
        s_shape,
        z_p_shape,
    ):
        (
            w,
            s,
            zp,
            clamp,
        ) = OVWeightCompressionAlgoBackend.get_compress_pipeline(
            weight_compression_parameter, w_shape, s_shape, z_p_shape, True
        )

        result = (clamp - zp) * s
        model = ov.Model([result], [w, s, zp])

        compiled_model = ov.compile_model(model)

        return lambda w, s, zp: compiled_model([w, s, zp])[0]

    @staticmethod
    def get_compress_pipeline(
        weight_compression_parameter: WeightCompressionParameters,
        w_shape,
        s_shape,
        z_p_shape,
        return_nodes=False,
    ):
        config = weight_compression_parameter.compression_config
        mode = config.mode
        assert mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]
        num_bits = config.num_bits

        level_low = 0
        level_high = 2**num_bits - 1

        w = opset.parameter(w_shape, name="w")
        s = opset.parameter(s_shape, name="s")
        zp = opset.parameter(z_p_shape, name="zp")

        result = opset.clamp(opset.round(w / s + zp), level_low, level_high, name="compressed_weights")

        if return_nodes:
            return w, s, zp, result

        model = ov.Model([result], [w, s, zp])

        compiled_model = ov.compile_model(model)

        return lambda w, s, zp: compiled_model([w, s, zp])[0]


class OVAWQAlgoAlgoBackend(OVWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return get_awq_patterns(om.OVMatMulMetatype, om.OVMultiplyMetatype)
