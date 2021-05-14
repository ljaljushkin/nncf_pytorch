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
import torch.nn.functional as F
from torch import nn

from nncf.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.dynamic_graph.graph_tracer import create_input_infos
from nncf.nncf_network import NNCFNetwork
# from nncf.progressive_shrinking import NASAlgoBuilder
from nncf.nas.bootstrapNAS.algo import BootstrapNASBuilder
from tests.helpers import BasicConvTestModel
from tests.helpers import get_empty_config


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class DynamicConv2d(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        # filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


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
    config = get_empty_config(input_sample_sizes=[20,30,7,7])
    model = BasicConvTestModel(in_channels=30, out_channels=20, kernel_size=7)
    input_info_list = create_input_infos(config)
    dummy_forward = create_dummy_forward_fn(input_info_list)

    compressed_model = NNCFNetwork(model, input_infos=input_info_list)
    composite_builder = BootstrapNASBuilder(config)
    composite_builder.apply_to(compressed_model)
    compression_ctrl = composite_builder.build_controller(compressed_model)

    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(5)
    output = dummy_forward(compressed_model)
    print(output.shape)

    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(3)
    output = dummy_forward(compressed_model)
    print(output.shape)

    # This should raise an exception
    for op in compression_ctrl.elastic_kernel_ops:
        op.set_active_kernel_size(9)
    output = dummy_forward(compressed_model)
    print(output.shape)


if __name__ == '__main__':
    test_elastic_kernel()