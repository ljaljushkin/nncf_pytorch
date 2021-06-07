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
from tests import test_models
from tests.helpers import BasicConvTestModel
from tests.helpers import get_empty_config
from tests.nas.test_nas_helpers import VGG11_K7

__all__ = [
    'test_elastic_width', 'test_elastic_width_vgg11_k7', 'test_elastic_kernel', 'test_elastic_kernel_bn', 'test_activate_subnet',
]

def _test_model(model_name):
    models = {
        'resnet18': [test_models.ResNet18(), [1, 3, 32, 32]],
        'vgg11': [test_models.VGG('VGG11'), [1, 3, 32, 32]],
        'vgg11_k7': [VGG11_K7(), [1, 3, 32, 32]], # for testing elastic kernel
    }
    model = models[model_name][0]
    config = get_empty_config(input_sample_sizes= models[model_name][1]) #[1, 3, 32, 32])
    input_info_list = create_input_infos(config)
    dummy_forward = create_dummy_forward_fn(input_info_list)

    compressed_model = NNCFNetwork(model, input_infos=input_info_list)
    config["test_mode"] = True
    composite_builder = BootstrapNASBuilder(config)
    composite_builder.apply_to(compressed_model)
    compression_ctrl = composite_builder.build_controller(compressed_model)

    return compressed_model, compression_ctrl, dummy_forward

def test_elastic_width():
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet18')

    # activate subnet-1
    num_filters_per_scope = {
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
        'ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv1]': 256,
        'ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv2]': 256,
        'ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[shortcut]/NNCFConv2d[0]': 256,
        'ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv1]': 256,
        'ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv2]': 256,
        'ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv1]': 512,
        'ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv2]': 512,
        'ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[shortcut]/NNCFConv2d[0]': 512,
        'ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv1]': 512,
        'ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv2]': 512
    }
    for scope, num_filters in num_filters_per_scope.items():
        op = compression_ctrl.scope_vs_elastic_op_map[scope]
        op.set_active_out_channels(num_filters)

    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

    # activate subnet-2
    for scope in list(num_filters_per_scope.keys())[:-5]:
        # op = compression_ctrl.scope_vs_elastic_width_op_map[scope]
        op = compression_ctrl.scope_vs_elastic_op_map[scope]
        op.set_active_out_channels(32)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

def test_elastic_width_vgg11_k7(): # VGG with Bias False
    compressed_model, compression_ctrl, dummy_forward = _test_model('vgg11_k7')  # _restnet18_test_model()

    # activate subnet-1
    num_filters_per_scope = {
        'VGG11_K7/Sequential[features]/NNCFConv2d[22]': 32,  #
    }
    for scope, num_filters in num_filters_per_scope.items():
        op = compression_ctrl.scope_vs_elastic_op_map[scope] # Unified op
        op.set_active_out_channels(num_filters)

    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

    # activate subnet-2
    for scope in list(num_filters_per_scope.keys())[:-5]:
        op = compression_ctrl.scope_vs_elastic_width_op_map[scope]
        op.set_active_out_channels(32)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

def test_elastic_kernel():
    config = get_empty_config(input_sample_sizes=[1, 30, 30, 7])
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

def test_elastic_width_bias(): #bias=True in conv2d
    compressed_model, compression_ctrl, dummy_forward = _test_model('vgg11')
    num_filters_per_scope = {
        'VGG/Sequential[features]/NNCFConv2d[22]': 32,  #
    }
    for scope, num_filters in num_filters_per_scope.items():
        op = compression_ctrl.scope_vs_elastic_op_map[scope] # Unified op
        op.set_active_out_channels(num_filters)

    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

def test_activate_subnet():
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet18')
    subnet_config = {'width': [64]}
    compression_ctrl.set_active_subnet(subnet_config)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]

