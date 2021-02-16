"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from functools import partial
from typing import Dict
from typing import List

from nncf.quantization.precision_init.base_init import BasePrecisionInitParams
from nncf.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup
from ..precision_constraints import HardwareQuantizationConstraints
from ...structures import QuantizationPrecisionInitArgs
from ...utils import in_scope_list


class ManualPrecisionInitParams(BasePrecisionInitParams):
    def __init__(self,
                 user_init_args: QuantizationPrecisionInitArgs = None,
                 bitwidth_per_scope: List[List] = None):
        super().__init__(user_init_args)
        self.bitwidth_per_scope = bitwidth_per_scope

    @classmethod
    def from_config(cls,
                    manual_init_params_dict: Dict):
        return cls(user_init_args=None,
                   bitwidth_per_scope=manual_init_params_dict.get("bitwidth_per_scope", []))


class ManualPrecisionInitializer(BasePrecisionInitializer):
    def __init__(self,
                 algo: 'ExperimentalQuantizationController',
                 params: ManualPrecisionInitParams,
                 hw_precision_constraints: HardwareQuantizationConstraints = None):
        super().__init__(algo, params, hw_precision_constraints)
        self._bitwidth_per_scope = params.bitwidth_per_scope

    def apply_init(self) -> SingleConfigQuantizerSetup:
        apply_bitwidth_to_scope_fn = self._apply_bitwidth_to_scope_for_pattern_based
        quantizer_setup = None
        if self._hw_precision_constraints:
            quantizer_setup = self._algo.get_quantizer_setup_for_current_state()
            apply_bitwidth_to_scope_fn = partial(self._apply_bitwidth_per_scope_for_propagation_based,
                                                 quantizer_setup=quantizer_setup)

        for pair in self._bitwidth_per_scope:
            bitwidth, scope_name = pair
            if not apply_bitwidth_to_scope_fn(bitwidth, scope_name):
                raise ValueError(
                    'Invalid scope name `{}`, failed to assign bitwidth {} to it'.format(scope_name, bitwidth))

        if not self._hw_precision_constraints:
            quantizer_setup = self._algo.get_quantizer_setup_for_current_state()
        return quantizer_setup

    def _apply_bitwidth_to_scope_for_pattern_based(self, bitwidth: int, scope_name: str) -> bool:
        is_matched = False
        for scope, quantizer in self._all_quantizers_per_scope.items():
            if in_scope_list(str(scope), scope_name):
                quantizer.num_bits = bitwidth
                is_matched = True
                break
        return is_matched

    def _apply_bitwidth_per_scope_for_propagation_based(self, bitwidth: int, scope_name: str,
                                                        quantizer_setup: SingleConfigQuantizerSetup):
        is_matched = False
        msg = 'Failed to assign bitwidth={} to `{}`,\n' \
              'because it is incompatible with supported quantization configs: {}'
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if str(qp.insertion_point) == scope_name:
                q_id = self._algo.setup_to_module_id_translation_dict[qp_id]
                q_configs = self._hw_precision_constraints.get(q_id)
                matched_q_configs = list(filter(lambda x: x.bits == bitwidth, q_configs))
                if not matched_q_configs:
                    raise ValueError(msg.format(bitwidth, scope_name, q_configs))
                qp.qconfig = matched_q_configs[0]
                is_matched = True
                break
        return is_matched
