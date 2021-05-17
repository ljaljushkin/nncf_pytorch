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

from nncf.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.dynamic_graph.graph_tracer import create_input_infos
from nncf.nas.bootstrapNAS.algo import BootstrapNASBuilder
from nncf.nncf_network import NNCFNetwork
from tests.helpers import BasicConvTestModel
from tests.helpers import get_empty_config


def test_elastic_width():
    # TODO: RuntimeError: running_mean should contain 3 elements not 64, should properly handle Conv + BatchNorm
    # config = get_empty_config(input_sample_sizes=[1, 3, 32, 32])

    # model = test_models.ResNet18()
    # num_classes = model.linear.out_features
    config = get_empty_config()
    model = BasicConvTestModel()
    # num_classes = 1
    # criterion = nn.MSELoss()
    # targets = torch.randn([num_classes])
    input_info_list = create_input_infos(config)
    dummy_forward = create_dummy_forward_fn(input_info_list)

    compressed_model = NNCFNetwork(model, input_infos=input_info_list)
    composite_builder = BootstrapNASBuilder(config)
    composite_builder.apply_to(compressed_model)
    compression_ctrl = composite_builder.build_controller(compressed_model)

    # final_subnet = BootstrapNASSearch(compressed_model, compression_ctrl,
    #                                   train_data_loader,
    #                                   criterion_fn,
    # or
    #                                   train_iteration_fn)

    # activate subnet-1
    for op in compression_ctrl.elastic_width_ops:
        op.set_active_out_channels(3)
    output = dummy_forward(compressed_model)
    print(output.shape)
    # optimizer.zero_grad()
    # loss = criterion(outputs, targets)
    # loss.backward()

    # activate subnet-2
    for op in compression_ctrl.elastic_width_ops:
        op.set_active_out_channels(7)
    output = dummy_forward(compressed_model)
    print(output.shape)
    # loss = criterion(outputs, targets)
    # optimizer.zero_grad()
    # loss.backward()


def test_elastic_kernel():
    config = get_empty_config(input_sample_sizes=[1, 30, 30, 7])
    model = BasicConvTestModel(in_channels=30, out_channels=20, kernel_size=7)

    input_info_list = create_input_infos(config)
    dummy_forward = create_dummy_forward_fn(input_info_list)

    compressed_model = NNCFNetwork(model, input_infos=input_info_list)
    composite_builder = BootstrapNASBuilder(config)
    composite_builder.apply_to(compressed_model)
    compression_ctrl = composite_builder.build_controller(compressed_model)

    print(compressed_model)

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


if __name__ == '__main__':
    test_elastic_kernel()
