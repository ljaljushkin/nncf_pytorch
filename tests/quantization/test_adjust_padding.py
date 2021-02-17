'''
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
from typing import List

import os
import pytest

from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import get_empty_config
from tests.quantization.test_precision_init import check_bitwidth_graph
from tests.test_compressed_graph import GeneralModelDesc
from tests.test_models.synthetic import MultiBranchesModel


class MultiBranchesModelDesc(GeneralModelDesc):
    NUM_WEIGHTS = 5
    NUM_ACTIVATIONS = 2

    def __init__(self, name: str):
        super().__init__(input_sample_sizes=[2, 3, 4, 4], model_name=name, model_builder=MultiBranchesModel)
        self._config = get_empty_config(input_sample_sizes=self.input_sample_sizes)
        self._config_update = {'compression': {'algorithm': 'quantization'}}

    @staticmethod
    def _get_scopes():
        w_scopes = [
            'MultiBranchesModel/NNCFConv2d[conv1]',
            'MultiBranchesModel/NNCFConv2d[conv2a]',
            'MultiBranchesModel/NNCFConv2d[conv2b]',
            'MultiBranchesModel/NNCFConv2d[conv2c]',
            'MultiBranchesModel/NNCFConv2d[conv2d]'
        ]
        a_scopes = [
            '/nncf_model_input_0',
            'MultiBranchesModel/NNCFConv2d[conv1]/conv2d_0'
        ]
        return w_scopes, a_scopes

    @staticmethod
    def _get_scope_hw():
        w_scopes = [
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv1]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv2a]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv2b]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv2c]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv2d]',
            'InsertionType.OPERATOR_POST_HOOK /nncf_model_input_0'
        ]
        a_scopes = [
            'InsertionType.OPERATOR_POST_HOOK MultiBranchesModel/NNCFConv2d[conv1]/conv2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/MaxPool2d[max_pool2b]/max_pool2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/NNCFConv2d[conv2a]/conv2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/NNCFConv2d[conv2c]/conv2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/NNCFConv2d[conv2d]/conv2d_0]'
        ]
        return w_scopes, a_scopes

    def trial(self, num_bits_for_weights: int = 8, num_bits_for_activations: int = 8):
        self._config_update['target_device'] = 'TRIAL'
        self._config_update['compression'].update(
            {
                'activations': {
                    'mode': 'symmetric',
                    'bits': num_bits_for_activations,
                    'per_channel': False,
                },
                'weights': {
                    'mode': 'symmetric',
                    'bits': num_bits_for_weights,
                    'per_channel': False,
                },
            })
        return self

    def vpu(self):
        self._config_update['target_device'] = 'VPU'
        return self

    def manual_precision(self, num_bits_for_weights: List[int], num_bits_for_activations: List[int], hw_config=False):
        scopes_factory = self._get_scope_hw if hw_config else self._get_scopes
        w_scopes, a_scopes = scopes_factory()
        bitwidth_per_scope = list(map(list, zip(num_bits_for_weights, w_scopes)))
        bitwidth_per_scope.extend(list(map(list, zip(num_bits_for_activations, a_scopes))))
        self._config_update['compression'].update(
            {'initializer': {'precision': {'type': 'manual', 'bitwidth_per_scope': bitwidth_per_scope}}})
        return self

    def get_config(self):
        self._config.update(self._config_update)
        self._config['compression'].update()
        return self._config


ADJUST_PAD_DESC_LIST = [
    MultiBranchesModelDesc(name="all_int4").trial(4, 4),
    MultiBranchesModelDesc(name="all_int8").trial(8, 8),
    MultiBranchesModelDesc(name="dw_int8").trial().manual_precision([4, 8, 4, 4, 4], [4, 4]),
    MultiBranchesModelDesc(name="all_weights_int8").trial(8, 4),
    MultiBranchesModelDesc(name="all_activations_int8").trial(4, 8),
    MultiBranchesModelDesc(name="vpu").vpu().manual_precision([4, 8, 4, 4, 4], [4, 4, 4, 4, 4], hw_config=True),
]


# TODO: try different branching merging strategy
# TODO: try custom HW config, like DWConv2d adjust pad only? for int4 only?
@pytest.mark.parametrize("desc", ADJUST_PAD_DESC_LIST,
                         ids=[m.model_name for m in ADJUST_PAD_DESC_LIST])
def test_adjust_padding_on_synthetic_models(desc: MultiBranchesModelDesc):
    model = desc.get_model()
    config = desc.get_config()

    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    check_bitwidth_graph(algo_ctrl, model, desc.get_dot_filename(), os.path.join('quantized', 'adjust_paddings'))
