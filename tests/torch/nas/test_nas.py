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
from collections import Callable
from typing import Tuple

import networkx as nx
import pytest
from torch.backends import cudnn
from torchvision.models import mobilenet_v2

from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASBuilder
from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASController
from nncf.torch.nas.bootstrapNAS.visualization import WidthGraph
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import manual_seed
from tests.torch import test_models
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import get_empty_config
from tests.torch.nas.test_nas_helpers import VGG11_K7


def _test_model(model_name) -> Tuple[NNCFNetwork, BootstrapNASController, Callable]:
    models = {
        'resnet50': [test_models.ResNet50(), [1, 3, 224, 224]],
        'resnet18': [test_models.ResNet18(), [1, 3, 224, 224]],
        'inception_v3': [test_models.Inception3(num_classes=10), [1, 3, 299, 299]],
        'vgg11': [test_models.VGG('VGG11'), [1, 3, 32, 32]],
        'vgg11_k7': [VGG11_K7(), [1, 3, 32, 32]],  # for testing elastic kernel
        'densenet_121': [test_models.DenseNet121(), [1, 3, 32, 32]],
        'mobilenet_v2': [mobilenet_v2(), [2, 3, 32, 32]]
    }
    model = models[model_name][0]
    print(model)
    config = get_empty_config(input_sample_sizes=models[model_name][1])
    config['compression'] = {'algorithm': 'bootstrapNAS'}
    input_info_list = create_input_infos(config)
    dummy_forward = create_dummy_forward_fn(input_info_list)

    compressed_model = NNCFNetwork(model, input_infos=input_info_list)
    print(compressed_model)
    config["test_mode"] = True
    composite_builder = BootstrapNASBuilder(config)
    composite_builder.apply_to(compressed_model)
    print(compressed_model)
    compression_ctrl = composite_builder.build_controller(compressed_model)

    return compressed_model, compression_ctrl, dummy_forward


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


def test_activate_subnet():
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet18')
    subnet_config = {'width': [64]}
    compression_ctrl.set_active_subnet(subnet_config)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]


def test_reactivate_supernet():
    # TODO
    pass


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
        pytest.skip('Skip test for MobileNet-v2 as ElasticDepthWise is not supported, Inception-V3 also fails on concat')

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



