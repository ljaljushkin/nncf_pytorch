import torch
import numpy as np

from tests.torch.test_models.resnet import ResNet18
from nncf.config import NNCFConfig
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.initialization import DefaultInitializingDataLoader, register_default_init_args
from nncf.torch.layers import NNCFConv2d
from tests.torch.helpers import create_ones_mock_dataloader
from nncf.torch.dynamic_graph.context import get_current_context

def get_basic_config():
    config = NNCFConfig()
    config.update(dict({
        "model": "basic_quant_conv",
        "input_info":
            {
                "sample_size": [1, 3, 224, 224],
            }
    }))
    return config

def test_skip_one_block_resnet18(mocker):
    model = ResNet18()
    nncf_config = get_basic_config()
    data_loader = DefaultInitializingDataLoader(create_ones_mock_dataloader(nncf_config))
    nncf_config = register_default_init_args(nncf_config, train_loader=data_loader)
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0',
                                     'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]/batch_norm_0']
    _, compressed_model = create_compressed_model(model, nncf_config)

    ctx = compressed_model._compressed_context
    spy_agent = mocker.spy(NNCFConv2d, "__call__")

    # we skipped first and second NNCFConv2d - modules
    ctx._elastic_depth = True # activate mode with elastic depth
    compressed_model(torch.ones([1, 3, 224, 224]))

    assert len(spy_agent.call_args_list) == 20
    assert id(spy_agent.call_args_list[2][0][1]) == id(spy_agent.call_args_list[1][0][1]) # TracedTensor


    spy_agent.reset_mock()

    ctx._elastic_depth = False 
    compressed_model(torch.ones([1, 3, 224, 224]))
    assert len(spy_agent.call_args_list) == 20
    assert id(spy_agent.call_args_list[2][0][1]) != id(spy_agent.call_args_list[1][0][1]) # TracedTensor


def test_skip_two_block_resnet18(mocker):
    model = ResNet18()
    nncf_config = get_basic_config()
    data_loader = DefaultInitializingDataLoader(create_ones_mock_dataloader(nncf_config))
    nncf_config = register_default_init_args(nncf_config, train_loader=data_loader)
    nncf_config['skipped_blocks'] = [
                                      ['ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0', # 1
                                       'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]/batch_norm_0'],
                                      ['ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0', # 8
                                       'ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2]/batch_norm_0']
                                    ]
    _, compressed_model = create_compressed_model(model, nncf_config)

    ctx = compressed_model._compressed_context
    spy_agent = mocker.spy(NNCFConv2d, "__call__")

    ctx._elastic_depth = True # activate mode with elastic depth
    compressed_model(torch.ones([1, 3, 224, 224]))

    assert len(spy_agent.call_args_list) == 20
    assert id(spy_agent.call_args_list[2][0][1]) == id(spy_agent.call_args_list[1][0][1]) # TracedTensor
    assert id(spy_agent.call_args_list[8][0][1]) == id(spy_agent.call_args_list[9][0][1]) # TracedTensor

    assert (spy_agent.call_args_list[2][0][1] == spy_agent.call_args_list[1][0][1]).sum() == np.prod(spy_agent.call_args_list[2][0][1].shape)
    spy_agent.reset_mock()

    ctx._elastic_depth = False 
    compressed_model(torch.ones([1, 3, 224, 224]))
    assert len(spy_agent.call_args_list) == 20

    assert id(spy_agent.call_args_list[2][0][1]) != id(spy_agent.call_args_list[1][0][1]) # TracedTensor
    assert id(spy_agent.call_args_list[8][0][1]) != id(spy_agent.call_args_list[9][0][1]) # TracedTensor