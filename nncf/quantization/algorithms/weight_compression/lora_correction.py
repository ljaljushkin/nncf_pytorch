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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from nncf.common.logging import nncf_logger
from nncf.common.utils.debug import DEBUG_LOG_DIR
from nncf.common.utils.debug import is_debug
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedLoraCorrectionParameters
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_fake_quantization_from_norm_weight
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType


class DebugInterface:
    """
    Utility class to collect and dump debug information of the Lora Correction algorithm.
    """

    def __init__(self):
        self._noise_per_layer = {}

    def add_noises(self, layer_name: str, value: float):
        self._noise_per_layer[layer_name] = value

    def dump_data(self):
        if not self._noise_per_layer:
            return
        dump_dir = Path(DEBUG_LOG_DIR) / "lora"
        dump_dir.mkdir(parents=True, exist_ok=True)

        layer_dir = dump_dir / "per_layer"
        layer_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._noise_per_layer)
        losses_path = dump_dir / "noises.csv"
        nncf_logger.debug("Quantization noise through the correction process is saved to: ", losses_path)
        df.to_csv(losses_path)

        for name in df.columns:
            plt.plot(df[name])
            filename = name.replace("/", "_") + ".jpg"
            plt.savefig(layer_dir / filename)
            plt.clf()

        delta = df.iloc[0] - df.iloc[-1]
        nncf_logger.debug("Is quantization noise reduced for all layers: ", all(delta > 0))

        _, ax = plt.subplots(1)
        ax.plot(delta)
        ax.set_xticklabels([])
        delta_path = dump_dir / "qnoise_change.jpg"
        nncf_logger.debug("Saving change in quantization noise for each layer to: ", delta_path)
        plt.savefig(delta_path)
        plt.clf()


class LoraCorrectionAlgorithm:
    """
    Contains implementation of LoRA Correction algorithm.

    The method reduces quantization noise after weight compression using low rank adapters.
    """

    def __init__(self, activations: Dict[str, List[Tensor]], lora_correction_params: AdvancedLoraCorrectionParameters):
        """
        :param activations: The input activations of the layers considered for compression.
        :param lora_correction_params: parameters to configure the algorithm.
        """
        self._activations = activations
        self._lora_correction_params = lora_correction_params
        self._debug_interface = DebugInterface() if is_debug() else None

    @property
    def is_int8_adapters(self) -> bool:
        return self._lora_correction_params.is_int8_adapters

    def is_applicable(self, wc_params: WeightCompressionParameters):
        return wc_params.compression_config.num_bits == 4

    def calculate_adapters(
        self, weight: Tensor, compressed_weight: Tensor, wc_params: WeightCompressionParameters
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculates low rank matrices for a given original and compressed weights.

        :param weight: original floating-point weight matrix.
        :param compressed_weight: compressed weight matrix.
        :param wc_params: parameters of weight compression.
        :return: two low rank matrices in the order of execution of corresponding linear layers.
        """
        try:
            adapters = self.calculate_adapters_fn(
                weight,
                compressed_weight,
                wc_params,
                self._lora_correction_params,
                self._activations,
                self._debug_interface,
            )
        finally:
            if self._debug_interface is not None:
                self._debug_interface.dump_data()
        return adapters

    @staticmethod
    def calculate_adapters_fn(
        weight: Tensor,
        compressed_weight: Tensor,
        wc_params: WeightCompressionParameters,
        lora_correction_params: AdvancedLoraCorrectionParameters,
        activations: Dict[str, List[Tensor]],
        debug_interface: Optional[DebugInterface] = None,
    ):
        """
        Calculates low rank matrices for a given original and compressed weights.
        The low rank matrices obtained by applying singular value decomposition (SVD) with lower rank for the
        difference between original weight and fake-quantized ones.
        Then, an iterative algorithm refines them. It solves a system of linear equations alternately fixing
        one matrix, then another.

        :param weight: original floating-point weight matrix.
        :param compressed_weight: compressed weight matrix.
        :param wc_params: parameters of weight compression.
        :param lora_correction_params: parameters to configure the algorithm.
        :param activations: The input activations of the layers considered for compression.
        :param debug_interface: utility class to collect and dump debug information, defaults to None
        :return: two low rank matrices in the order of execution of corresponding linear layers.
        """
        rank, n_iters, w_regularization, subset_size = (
            lora_correction_params.rank,
            lora_correction_params.n_iters,
            lora_correction_params.w_regularization,
            lora_correction_params.subset_size,
        )
        layer_name = wc_params.node_with_weight.node_name
        mode = wc_params.compression_config.mode
        reduction_axis = wc_params.reduction_axes[0] if wc_params.compression_config.group_size != -1 else -1
        if mode in (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM):
            fq_weights = do_dequantization(
                compressed_weight.tensor,
                compressed_weight.scale,
                compressed_weight.zero_point,
                reduction_axis,
            )
        elif mode == CompressWeightsMode.NF4:
            fq_weights = do_nf4_fake_quantization_from_norm_weight(
                compressed_weight.tensor,
                compressed_weight.scale,
                reduction_axis,
            )
        else:
            raise ValueError(
                f"{mode.value} mode is invalid for Lora Correction algorithm. Supported modes: INT4_SYM, INT4_ASYM, NF4"
            )
        # fq_w + residual = w   =>  residual = w - fq_w
        # O stands for output dimension, H - input dimension or hidden size, SS - samples size, R - rank.
        svd_residual = fns.astype(weight - fq_weights, TensorDataType.float32)  # [O, H]
        if wc_params.reduction_axes == 0:
            # TODO: always [O, H] or [H, O] ???
            # TODO: clone afterwards ???
            # in which cases its needed??
            svd_residual = fns.transpose(svd_residual)

        residual = svd_residual.clone()  # [O, H] ??? after transpose??

        # TODO: is it always has input dimension on the first position?
        s, X = process_stats(activations[layer_name], subset_size)  # [H], [H, SS]

        if wc_params.compression_config.group_size > 0:
            # Multiply residual of weights by maximum channel magnitude of activations normalized per quantization
            # group. As a consequence, weights corresponding to a "noisy" activations has a higher error to correct.
            # Empirically, it leads to a better accuracy.
            gs = wc_params.compression_config.group_size
            n_gs = s.shape[0] // gs
            for i in range(n_gs):
                offset = i * gs
                denum = fns.sum(s[offset : offset + gs])
                s[offset : offset + gs] = s[offset : offset + gs] / denum
                denum = fns.max(s[offset : offset + gs])
                s[offset : offset + gs] = s[offset : offset + gs] / denum
            s = fns.expand_dims(s, 0)  # [1, H]
            svd_residual = svd_residual * s  # [O, H]

        # Low-rank approximation.
        U_full, S_full, V_full = fns.linalg.svd(svd_residual, full_matrices=False)
        U = U_full[:, :rank]  # [O, R]
        S = fns.diag(S_full[:rank])  # [R, R]
        V = V_full[:rank, :]  # [R, H]
        V = S @ V  # [R, H]

        # An iterative algorithm for correction (refinement) of the low-rank adapters.

        mean_noises = []
        noise = residual @ X  # [O, SS]
        for i in range(n_iters):
            # Part 1: V is fixed, find U.
            VX = V @ X  # [R, SS]
            if not w_regularization:
                # 1) U @ V @ X = noise
                # (V @ X).t @ U.t = noise.t  ---> a @ x = b
                sol = fns.linalg.lstsq(
                    fns.transpose(VX), fns.transpose(noise), driver="gelsy"
                )  # [SS, R]*[R, O]=[SS, O]
            else:
                # 1) U @ V = res         <--- regularization
                # 2) U @ V @ X = noise
                # U @ |V V @ X| = |res noise|
                VVX = fns.concatenate([V, VX], axis=1)  # [R, H + SS]
                noiseR = fns.concatenate([residual, noise], axis=1)  # [O, H + SS]
                sol = fns.linalg.lstsq(fns.transpose(VVX), fns.transpose(noiseR), driver="gelsy")
            if debug_interface is not None and i == 0:
                init_noise = noise
                mean_noise_before_svd = fns.mean(fns.abs(init_noise)).item()
                mean_noise_after_svd = fns.mean(fns.abs(init_noise - (U @ V) @ X)).item()
                mean_noises.extend([mean_noise_before_svd, mean_noise_after_svd])
            U = fns.transpose(sol)
            if debug_interface is not None:
                mean_noise_after_correct_U = fns.mean(fns.abs(init_noise - (U @ V) @ X)).item()
                mean_noises.append(mean_noise_after_correct_U)
                nncf_logger.debug(
                    f"{i} Correction U: ", mean_noise_before_svd, mean_noise_after_svd, mean_noise_after_correct_U
                )

            # Part 2: U is fixed, find V.
            UI = fns.linalg.pinv(U)  # [R, O]
            noiseU = UI @ noise  # [R, SS]
            if not w_regularization:
                # UI = U^-1
                # 1) U @ V @ X = noise
                # 1) V @ X = UI @ noise
                # X.t @ V.t = (UI @ noise).t  ---> a @ x = b
                sol = fns.linalg.lstsq(fns.transpose(X), fns.transpose(noiseU), driver="gelsy")
            else:
                # UI = U^-1, E - identity matrix
                # 1) U @ V = res           <--- regularization
                # 1) V @ E = UI @ res
                # 2) U @ V @ X = noise
                # 2) V @ X = UI @ noise
                # V @ |E X| = | (UI @ res) (UI @ noise) |
                E = fns.eye(V.shape[1], backend=V.backend, dtype=V.dtype)  # [H, H]
                IX = fns.concatenate([E, X], axis=1)  # [H, H + SS]
                wU = UI @ residual  # [R, H]
                noiseR = fns.concatenate([wU, noiseU], axis=1)  # [R, H + SS]
                sol = fns.linalg.lstsq(fns.transpose(IX), fns.transpose(noiseR), driver="gelsy")
            V = fns.transpose(sol)
            if debug_interface is not None:
                mean_noise_after_correct_V = fns.mean(fns.abs(init_noise - (U @ V) @ X)).item()
                mean_noises.append(mean_noise_after_correct_V)
                nncf_logger.debug(
                    f"{i} Correction V: ", mean_noise_before_svd, mean_noise_after_svd, mean_noise_after_correct_V
                )
                debug_interface.add_noises(layer_name, mean_noises)
        return V, U
