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

import networkx as nx
import pytest
import torch
from torch import nn
from torch.backends import cudnn

from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASController
from nncf.torch.nas.bootstrapNAS.visualization import WidthGraph
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import manual_seed
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config
from tests.torch.nas.test_nas import _test_model


def test_can_set_elastic_width_get_elastic_parameters(tmp_path):
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet18')
    # activate subnet-1
    layer_name_vs_width_map = {
        'ResNet/NNCFConv2d[conv1]': 32,  # 64,
        'ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv1]': 32,  # 64
        'ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv2]': 32,  # 64
        'ResNet/Sequential[layer1]/BasicBlock[1]/NNCFConv2d[conv1]': 32,  # 64
        'ResNet/Sequential[layer1]/BasicBlock[1]/NNCFConv2d[conv2]': 32,  # 64
        'ResNet/Sequential[layer2]/BasicBlock[0]/NNCFConv2d[conv1]': 64,  # 128
        'ResNet/Sequential[layer2]/BasicBlock[0]/NNCFConv2d[conv2]': 64,  # 128
        'ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[shortcut]/NNCFConv2d[0]': 64,  # 128
        'ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv1]': 64,  # 128
        'ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv2]': 64,  # 128
        'ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv1]': 128,  # 256
        'ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv2]': 128,  # 256
        'ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[shortcut]/NNCFConv2d[0]': 128,  # 256
        'ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv1]': 128,  # 256,
        'ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv2]': 128,  # 256,
        'ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv1]': 512,
        # TODO: can't be pruned due to model analysis (because of AvgPool and Linear)
        # 'ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv2]': 512,
        # 'ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[shortcut]/NNCFConv2d[0]': 512,
        'ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv1]': 512,
        # 'ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv2]': 512
    }

    visualize_width(compressed_model, compression_ctrl, tmp_path / 'original.dot')
    dummy_forward(compressed_model)

    compression_ctrl.activate_elastic_width_configuration_by_layer_names(layer_name_vs_width_map)
    visualize_width(compressed_model, compression_ctrl, tmp_path / 'subnet.dot')
    dummy_forward(compressed_model)


@pytest.fixture
def _seed():
    cudnn.deterministic = True
    cudnn.benchmark = False
    manual_seed(0)


def visualize_width(model, ctrl, path):
    nx_graph = WidthGraph(model, ctrl).get()
    nx.drawing.nx_pydot.write_dot(nx_graph, path)


@pytest.mark.parametrize('model_name', [
    'resnet18', 'resnet50', 'densenet_121', 'mobilenet_v2', 'vgg11', 'inception_v3'
])
class TestModelScope:
    def test_can_sample_random_elastic_width_configurations(self, _seed, tmp_path, model_name):
        if model_name in ['inception_v3']:
            pytest.skip(
                'Skip test for Inception-V3 as it fails because of 2 issues: '
                'not able to set DynamicInputOp to train-only layers (60976) and '
                'invalid padding update in elastic kernel (60990)')

        compressed_model, compression_ctrl, dummy_forward = _test_model(model_name)
        compressed_model.eval()
        visualize_width(compressed_model, compression_ctrl, tmp_path / f'{model_name}_original.dot')
        N = 5
        for i in range(N):
            cluster_id_vs_width_map = compression_ctrl._get_random_width_conf()
            print('Set random width configuration: {}'.format(cluster_id_vs_width_map.values()))
            compression_ctrl.activate_elastic_width_configuration_by_cluster_id(cluster_id_vs_width_map)
            visualize_width(compressed_model, compression_ctrl, tmp_path / f'{model_name}_random_{i}.dot')
            dummy_forward(compressed_model)

    def test_weight_reorg(self, model_name, _seed):
        if model_name in ['inception_v3']:
            pytest.skip('Skip test for Inception-V3 because of invalid padding update in elastic kernel (60990)')

        compressed_model, compression_ctrl, dummy_forward = _test_model(model_name)
        compressed_model.eval()
        before_reorg = dummy_forward(compressed_model)
        compression_ctrl.reorganize_weights()
        after_reorg = dummy_forward(compressed_model)
        # TODO(nlyalyus): should tolerance be less than 1e-2?
        compare_vectors_ignoring_the_order(after_reorg, before_reorg, atol=1e-2)


class TwoSequentialConvBNTestModel(nn.Module):
    """
    conv1 -> bn1 -> conv2 -> bn2
    """
    INPUT_SIZE = [1, 1, 1, 1]

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 1)
        self.bn1 = nn.BatchNorm2d(3)

        bias = torch.Tensor([3, 1, 2])
        weights = bias.reshape(3, 1, 1, 1)
        self.set_params(bias, weights, self.conv1, self.bn1)

        self.conv2 = create_conv(3, 2, 1)
        self.bn2 = nn.BatchNorm2d(3)

        weight = torch.Tensor([[1, 2, 0],
                               [2, 3, 0]]).reshape(2, 3, 1, 1)
        bias = torch.Tensor([1, 2])
        self.set_params(bias, weight, self.conv2, self.bn2)

        self.all_layers = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2, nn.ReLU())

    @staticmethod
    def set_params(bias, weight, conv, bn):
        conv.weight.data = weight
        list_params = TwoSequentialConvBNTestModel.get_bias_like_params(bn, conv)
        for param in list_params:
            param.data = bias

    @staticmethod
    def compare_params(bias, weight, conv, bn):
        assert torch.allclose(conv.weight, weight)
        list_params = TwoSequentialConvBNTestModel.get_bias_like_params(bn, conv)
        for param in list_params:
            assert torch.allclose(param, bias)

    @staticmethod
    def get_bias_like_params(bn, conv):
        list_params = [conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var]
        return list_params

    def check_reorg(self):
        ref_bias_1 = torch.Tensor([3, 2, 1])
        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1)
        ref_bias_2 = torch.Tensor([2, 1])
        ref_weights_2 = torch.Tensor([[2, 0, 3],
                                      [1, 0, 2]]).reshape(2, 3, 1, 1)
        TwoSequentialConvBNTestModel.compare_params(ref_bias_1, ref_weights_1, self.conv1, self.bn1)
        TwoSequentialConvBNTestModel.compare_params(ref_bias_2, ref_weights_2, self.conv2, self.bn2)

    def forward(self, x):
        return self.all_layers(x)


class TwoConvAddConvTestModel(nn.Module):
    """
    conv1  conv2
        \ /
        add
    """
    INPUT_SIZE = [1, 1, 1, 1]

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 1)
        self.conv2 = create_conv(1, 3, 1)
        self._init_params(self.conv1, torch.Tensor([3, 1, 2]))
        self._init_params(self.conv2, torch.Tensor([1, 2, 4]))
        self.all_layers = nn.Sequential(self.conv1, self.conv2)

    @staticmethod
    def _init_params(conv, data):
        conv.bias.data = data
        weight = data.reshape(3, 1, 1, 1)
        conv.weight.data = weight

    @staticmethod
    def compare_params(bias, weight, conv, bn):
        assert torch.allclose(conv.weight, weight)
        list_params = TwoSequentialConvBNTestModel.get_bias_like_params(bn, conv)
        for param in list_params:
            assert torch.allclose(param, bias)

    @staticmethod
    def get_bias_like_params(bn, conv):
        list_params = [conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var]
        return list_params

    def check_reorg(self):
        ref_bias_1 = torch.Tensor([2, 3, 1])
        ref_bias_2 = torch.Tensor([4, 1, 2])

        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1)
        ref_weights_2 = ref_bias_2.reshape(3, 1, 1, 1)

        assert torch.allclose(self.conv1.weight, ref_weights_1)
        assert torch.allclose(self.conv1.bias, ref_bias_1)
        assert torch.allclose(self.conv2.weight, ref_weights_2)
        assert torch.allclose(self.conv2.bias, ref_bias_2)

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


def compare_vectors_ignoring_the_order(vec1: torch.Tensor, vec2: torch.Tensor, rtol=1e-05, atol=1e-08):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    if len(vec1.shape) > 1 or len(vec2.shape) > 1:
        raise ValueError('Tensor is expected to have only one dimension not equal to 1')
    assert torch.allclose(torch.sort(vec1)[0], torch.sort(vec2)[0], rtol=rtol, atol=atol)


@pytest.mark.parametrize('model_cls', (TwoConvAddConvTestModel, TwoSequentialConvBNTestModel))
def test_width_reorg_for_basic_models(model_cls):
    model = model_cls()
    config = get_empty_config(input_sample_sizes=model_cls.INPUT_SIZE)
    config['compression'] = {'algorithm': 'bootstrapNAS'}
    config["test_mode"] = True
    model, ctrl = create_compressed_model_and_algo_for_test(model, config)  # type: NNCFNetwork, BootstrapNASController

    model.eval()
    dummy_input = torch.Tensor([1]).reshape(model_cls.INPUT_SIZE)
    before_reorg = model(dummy_input)
    ctrl.reorganize_weights()
    after_reorg = model(dummy_input)

    compare_vectors_ignoring_the_order(after_reorg, before_reorg)
    model.get_nncf_wrapped_model().check_reorg()
