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

from torchvision.models import mobilenet_v2

from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASBuilder
from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASController
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch import test_models
from tests.torch.helpers import get_empty_config
from tests.torch.nas.test_nas_helpers import VGG11_K7


def _test_model(model_name) -> Tuple[NNCFNetwork, BootstrapNASController, Callable]:
    models = {
        'resnet50': [test_models.ResNet50(), [1, 3, 32, 32]],
        'resnet18': [test_models.ResNet18(), [1, 3, 32, 32]],
        'inception_v3': [test_models.Inception3(num_classes=10), [1, 3, 299, 299]],
        'vgg11': [test_models.VGG('VGG11'), [1, 3, 32, 32]],
        'vgg11_k7': [VGG11_K7(), [1, 3, 32, 32]],  # for testing elastic kernel
        'densenet_121': [test_models.DenseNet121(), [1, 3, 32, 32]],
        'mobilenet_v2': [mobilenet_v2(), [1, 3, 32, 32]],
    }
    model = models[model_name][0]
    print(model)
    config = get_empty_config(input_sample_sizes=models[model_name][1])
    config['compression'] = {'algorithm': 'bootstrapNAS'}
    config['input_info'][0].update({'filler': 'random'})
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


def test_activate_subnet():
    compressed_model, compression_ctrl, dummy_forward = _test_model('resnet18')
    subnet_config = {'width': [64]}
    compression_ctrl.set_active_subnet(subnet_config)
    output = dummy_forward(compressed_model)
    assert list(output.shape) == [1, 10]


def test_reactivate_supernet():
    # TODO
    pass

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