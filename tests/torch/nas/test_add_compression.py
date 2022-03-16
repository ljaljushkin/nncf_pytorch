"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Tuple

import torch

from nncf import NNCFConfig
from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.visualization import SubnetGraph
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_algorithm_builder
from nncf.torch.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.torch.model_creation import create_compression_algorithm_builder
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.descriptors import THREE_CONV_TEST_DESC
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.models.synthetic import ThreeConvModelMode
from tests.torch.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.torch.test_compressed_graph import check_nx_graph


def add_compression(nncf_network: NNCFNetwork,
                    compression_ctrl,
                    compression_config: NNCFConfig) -> Tuple[NNCFNetwork, BaseController]:
    """
    Add compression to existing compressed model and compression controller

    :param nncf_network: NNCFNetwork that can contain some compression operations
    :param compression_ctrl: compression controller that corresponds to the compression in the given NNCFNetwork
    :param compression_config: a configuration of compression to be added
    :return:
    """
    old_compression_state = compression_ctrl.get_compression_state()
    # TODO: training_ctrl requires bn_adapt_args on resume, but old config is empty
    old_builder = resume_compression_algorithm_builder(old_compression_state)

    new_model = nncf_network.get_clean_shallow_copy()

    new_builder = create_compression_algorithm_builder(compression_config, should_init=True)
    composite_builder = PTCompositeCompressionAlgorithmBuilder.from_builders([old_builder, new_builder])
    model = composite_builder.apply_to(new_model)
    new_ctrl = new_builder.build_controller(new_model)
    new_ctrl.load_state(old_compression_state[BaseController.CONTROLLER_STATE])
    return model, new_ctrl


def test_can_add_quantization():
    ref_model = THREE_CONV_TEST_DESC.model_creator()
    device = move_model_to_cuda_if_available(ref_model)
    ref_model.to(device)
    input_ = torch.ones(ref_model.INPUT_SIZE).to(device)
    ref_model.mode = ThreeConvModelMode.SUPERNET
    ref_supernet_output = ref_model(input_)

    nas_model, nas_train_ctrl = create_bnas_model_and_ctrl_by_test_desc(THREE_CONV_TEST_DESC, mode='manual')
    elasticity_ctrl = nas_train_ctrl.elasticity_controller
    actual_output = nas_model(input_)
    assert torch.equal(actual_output, ref_supernet_output)
    print(nas_model)
    nas_model.rebuild_graph()
    dot_file = 'three_conv_nas.dot'
    subnet_graph = SubnetGraph(nas_model.get_graph(), nas_train_ctrl.multi_elasticity_handler).get()
    check_nx_graph(subnet_graph, dot_file, 'nas')

    quantization_config = get_quantization_config_without_range_init(THREE_CONV_TEST_DESC[2])
    register_bn_adaptation_init_args(quantization_config)

    int8_nas_model, int8_nas_train_ctrl = add_compression(nas_model, elasticity_ctrl, quantization_config)
    print(int8_nas_model)
    int8_output = int8_nas_model(input_)
    assert not torch.equal(int8_output, ref_supernet_output)
    dot_file = 'three_conv_nas_int8.dot'
    subnet_graph = SubnetGraph(int8_nas_model.get_graph(), nas_train_ctrl.multi_elasticity_handler).get()
    check_nx_graph(subnet_graph, dot_file, 'nas')
    dot_file = 'three_conv_nas_int8_full.dot'
    int8_nas_model.rebuild_graph()
    check_nx_graph(int8_nas_model.get_graph().get_graph_for_structure_analysis(), dot_file, 'nas')
