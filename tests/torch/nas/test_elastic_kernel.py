"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASBuilder
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import get_empty_config
from tests.torch.nas.test_nas import _test_model


def test_elastic_kernel():
    config = get_empty_config(input_sample_sizes=[1, 30, 30, 7])
    config['compression'] = {'algorithm': 'bootstrapNAS'}
    config["test_mode"] = True
    model = BasicConvTestModel(in_channels=30, out_channels=20, kernel_size=7)

    input_info_list = create_input_infos(config)
    dummy_forward = create_dummy_forward_fn(input_info_list)

    compressed_model = NNCFNetwork(model, input_infos=input_info_list)
    composite_builder = BootstrapNASBuilder(config)
    composite_builder.apply_to(compressed_model)
    compression_ctrl = composite_builder.build_controller(compressed_model)

    print("Kernel size 7")
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 20, 30, 7]

    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(5)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 20, 30, 7]

    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(3)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 20, 30, 7]

    # This should raise an exception
    # for op in compression_ctrl.elastic_kernel_ops:
    #     op.set_active_kernel_size(9)
    # output = dummy_forward(compressed_model)


def test_elastic_kernel_bn():
    compressed_model, compression_ctrl, dummy_forward = _test_model('vgg11_k7')  # _restnet18_test_model()
    print(compressed_model)
    print("Kernel size 7")
    output = dummy_forward(compressed_model)
    print(output)
    assert list(output.shape) == [1, 10]

    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(5)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(3)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]
