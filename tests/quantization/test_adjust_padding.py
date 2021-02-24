"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import List

import os
import pytest

from nncf.hw_config import HWConfig
from nncf.quantization.quantizer_propagation import PropagationStrategy
from nncf.quantization.quantizer_propagation import QuantizerPropagationSolver
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import get_empty_config
from tests.quantization.test_hawq_precision_init import check_bitwidth_graph
from tests.test_compressed_graph import GeneralModelDesc
from tests.test_helpers import load_exported_onnx_version
from tests.test_models.synthetic import MultiBranchesModel


class MultiBranchesModelDesc(GeneralModelDesc):
    NUM_WEIGHTS = 5
    NUM_ACTIVATIONS = 2

    def __init__(self, name: str):
        super().__init__(input_sample_sizes=[2, 3, 4, 4], model_name=name, model_builder=MultiBranchesModel)
        self._config = get_empty_config(input_sample_sizes=self.input_sample_sizes)
        self._config_update = {'compression': {'algorithm': 'quantization'}}
        self._hw_config = False
        self.custom_hw_config_dict = None
        self.propagation_strategy = PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT

    def requant_prop_strategy(self):
        self.propagation_strategy = PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION
        return self

    @staticmethod
    def _get_scopes():
        w_scopes = [
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv_a]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv_b]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv_c]',
            'InsertionType.NNCF_MODULE_PRE_OP MultiBranchesModel/NNCFConv2d[conv_d]',
        ]
        a_scopes = [
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/NNCFConv2d[conv_a]/conv2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/MaxPool2d[max_pool_b]/max_pool2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/NNCFConv2d[conv_c]/conv2d_0',
            'InsertionType.OPERATOR_PRE_HOOK 0 MultiBranchesModel/NNCFConv2d[conv_d]/conv2d_0'
        ]
        return w_scopes, a_scopes

    def trial(self, num_bits_for_weights: int = 8, num_bits_for_activations: int = 8):
        self._config_update['target_device'] = 'TRIAL'
        trial_config = {
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
        }
        self._config_update['compression'].update(trial_config)
        return self

    def vpu(self):
        self._config_update['target_device'] = 'VPU'
        return self

    def custom_hw(self):
        custom_hw_config_dict = {
            "target_device": "VPU",
            "config": {
                "quantization": {
                    "q4": {
                        "bits": 4,
                        "mode": "symmetric",
                        "granularity": "pertensor"
                    },
                }
            },
            "operations": [
                {
                    "type": "Convolution",
                    "quantization": {
                        "activations": "q4",
                        "weights": "q4"
                    }
                },
                {
                    "type": "DepthWiseConvolution",
                    "attributes": {
                        "adjust_padding": True
                    },
                    "quantization": {
                        "activations": "q4",
                        "weights": "q4"
                    }
                },
            ]
        }
        self.custom_hw_config_dict = custom_hw_config_dict
        return self

    def manual_precision(self, num_bits_for_weights: List[int], num_bits_for_activations: List[int]):
        scopes_factory = self._get_scopes
        w_scopes, a_scopes = scopes_factory()
        bitwidth_per_scope = list(map(list, zip(num_bits_for_weights, w_scopes)))
        bitwidth_per_scope.extend(list(map(list, zip(num_bits_for_activations, a_scopes))))
        init_config = {'initializer': {'precision': {'type': 'manual', 'bitwidth_per_scope': bitwidth_per_scope}}}
        self._config_update['compression'].update(init_config)
        return self

    def get_config(self):
        self._config.update(self._config_update)
        self._config['compression'].update()
        return self._config


ADJUST_PAD_DESC_LIST = [
    MultiBranchesModelDesc(name="vpu_all_int8").vpu(),
    MultiBranchesModelDesc(name="vpu_all_weights_int8").vpu().manual_precision([8, 8, 8, 8], [8, 4, 4, 4]),
    MultiBranchesModelDesc(name="vpu_all_activations_int8").vpu().manual_precision([8, 4, 4, 4], [8, 8, 8, 4]),
    MultiBranchesModelDesc(name="vpu_bd_int8").vpu().manual_precision([4, 4, 4, 4], [8, 8, 4, 8]),
    MultiBranchesModelDesc(name="vpu_max_int4").vpu().manual_precision([4, 4, 4, 4], [8, 4, 4, 4]),
    MultiBranchesModelDesc(name="vpu_all_int8_requnt").vpu().requant_prop_strategy(),
    MultiBranchesModelDesc(name="vpu_all_weights_int8_requnt").vpu().
        manual_precision([8, 8, 8, 8], [8, 4, 4, 4]).requant_prop_strategy(),
    MultiBranchesModelDesc(name="vpu_all_activations_int8_requnt").vpu().
        manual_precision([8, 4, 4, 4], [8, 8, 8, 4]).requant_prop_strategy(),
    MultiBranchesModelDesc(name="vpu_bd_int8_requnt").vpu().
        manual_precision([4, 4, 4, 4], [8, 8, 4, 8]).requant_prop_strategy(),
    MultiBranchesModelDesc(name="vpu_max_int4_requnt").vpu().
        manual_precision([4, 4, 4, 4], [8, 4, 4, 4]).requant_prop_strategy(),
    MultiBranchesModelDesc(name="custom").custom_hw()
]


@pytest.mark.parametrize("desc", ADJUST_PAD_DESC_LIST,
                         ids=[m.model_name for m in ADJUST_PAD_DESC_LIST])
def test_adjust_padding_on_synthetic_models(desc: MultiBranchesModelDesc, mocker, monkeypatch):
    model = desc.get_model()
    config = desc.get_config()

    if desc.custom_hw_config_dict:
        hw_config_from_json = mocker.patch('nncf.hw_config.HWConfig.from_json')
        hw_config_from_json.return_value = HWConfig.from_dict(desc.custom_hw_config_dict)

    monkeypatch.setattr(QuantizerPropagationSolver, 'DEFAULT_PROPAGATION_STRATEGY', desc.propagation_strategy)

    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    check_bitwidth_graph(algo_ctrl, model, desc.get_dot_filename(), os.path.join('quantized', 'adjust_paddings'))


def test_onnx_export_to_fake_quantize_with_adjust_pad(tmp_path):
    desc = MultiBranchesModelDesc(name="vpu_max_int4").vpu().manual_precision([4, 4, 4, 4], [8, 4, 4, 4])
    model = desc.get_model()
    nncf_config = desc.get_config()

    onnx_model_proto = load_exported_onnx_version(nncf_config, model,
                                                  path_to_storage_dir=tmp_path)
    num_fq = 0
    num_model_nodes = 0
    num_adjust_pad_nodes = 0
    num_other_nodes = 0
    # pylint:disable=no-member
    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == 'FakeQuantize':
            num_fq += 1
        elif op_type in ['Conv', 'Constant', 'Relu', 'MaxPool']:
            num_model_nodes += 1
        elif op_type in ['Pad']:
            pad_value_attr = node.attribute[2]
            assert pad_value_attr.f == 0.5
            num_adjust_pad_nodes += 1
        else:
            num_other_nodes += 1
            print(op_type)
    assert num_fq == 8
    assert num_other_nodes == 0
