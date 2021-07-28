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

from nncf.torch.nas.bootstrapNAS.visualization import WidthGraph
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import manual_seed
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
def test_can_sample_random_elastic_width_configurations(_seed, tmp_path, model_name):
    if model_name in ['mobilenet_v2', 'inception_v3']:
        pytest.skip(
            'Skip test for MobileNet-v2 as ElasticDepthWise is not supported, Inception-V3 also fails on concat')

    compressed_model, compression_ctrl, dummy_forward = _test_model(model_name)
    visualize_width(compressed_model, compression_ctrl, tmp_path / f'{model_name}_original.dot')
    N = 10
    for i in range(N):
        cluster_id_vs_width_map = compression_ctrl._get_random_width_conf()
        print('Set random width configuration: {}'.format(cluster_id_vs_width_map.values()))
        compression_ctrl.activate_elastic_width_configuration_by_cluster_id(cluster_id_vs_width_map)
        visualize_width(compressed_model, compression_ctrl, tmp_path / f'{model_name}_random_{i}.dot')
        dummy_forward(compressed_model)


def test_restore_supernet_from_checkpoint(tmp_path):
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet18')
    compression_ctrl.main_path = tmp_path
    compression_ctrl.config['fine_tuner'] = 'progressive_shrinking'
    compression_ctrl.save_supernet_checkpoint(checkpoint_name='test', epoch=-1)
    import torch
    print(f'{tmp_path}/test.pth')
    supernet = torch.load(f'{tmp_path}/test.pth')
    print(supernet.keys())
    assert supernet['fine_tuner'] == 'progressive_shrinking'
    # compression_ctrl.load_supernet_checkpoint()


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def manual_weight_reorganization(compressed_model: NNCFNetwork):
    bottlenecks = [2, 3, 5, 2]
    for layer in range(1, 5):
        for bottleneck in range(1, bottlenecks[layer - 1] + 1):
            from nncf.torch.dynamic_graph.scope import Scope

            conv3 = compressed_model.get_module_by_scope(
                Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[conv3]'))
            conv2 = compressed_model.get_module_by_scope(
                Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[conv2]'))
            bn2 = compressed_model.get_module_by_scope(
                Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[bn2]'))
            conv1 = compressed_model.get_module_by_scope(
                Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[conv1]'))
            bn1 = compressed_model.get_module_by_scope(
                Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[bn1]'))

            # conv3 -> conv2
            importance = torch.sum(torch.abs(conv3.weight.data), dim=(0, 2, 3))
            sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
            conv3.weight.data = torch.index_select(conv3.weight.data, 1, sorted_idx)
            adjust_bn_according_to_idx(bn2, sorted_idx)
            conv2.weight.data = torch.index_select(conv2.weight.data, 0, sorted_idx)

            # conv2 -> conv1
            importance = torch.sum(torch.abs(conv2.weight.data), dim=(0, 2, 3))
            sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

            conv2.weight.data = torch.index_select(conv2.weight.data, 1, sorted_idx)
            adjust_bn_according_to_idx(bn1, sorted_idx)
            conv1.weight.data = torch.index_select(conv1.weight.data, 0, sorted_idx)


def test_weight_reorg():
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet50')
    manual_weight_reorganization(compressed_model)
    dummy_forward(compressed_model)
