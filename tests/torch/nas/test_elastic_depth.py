import torch
import numpy as np
import onnx
import onnxruntime as rt

from tests.torch.test_models.resnet import ResNet18, ResNet50
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
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/BasicBlock[0]/RELU_0', # 1
                                     'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]/batch_norm_0']
    _, compressed_model = create_compressed_model(model, nncf_config)

    ctx = compressed_model._compressed_context
    spy_agent_conv2d = mocker.spy(NNCFConv2d, "__call__")
    spy_agent_bn = mocker.spy(torch.nn.BatchNorm2d, "__call__")

    ctx._elastic_depth = True # activate mode with elastic depth
    ctx.set_active_skipped_block([0])
    compressed_model(torch.ones([1, 3, 224, 224]))

    assert id(spy_agent_conv2d.call_args_list[2][0][1]) == id(spy_agent_bn.call_args_list[2][0][1]) # TracedTensor
    
    # check torch.tensor.data by element
    assert (spy_agent_conv2d.call_args_list[2][0][1] == spy_agent_bn.call_args_list[2][0][1]).sum() == np.prod(spy_agent_bn.call_args_list[2][0][1].shape)

    spy_agent_conv2d.reset_mock()
    spy_agent_bn.reset_mock()

    ctx._elastic_depth = False 
    compressed_model(torch.ones([1, 3, 224, 224]))
    assert id(spy_agent_conv2d.call_args_list[2][0][1]) != id(spy_agent_bn.call_args_list[2][0][1]) # TracedTensor


def test_can_export_model_with_one_skipped_block_resnet18(mocker):
    model = ResNet18()
    nncf_config = get_basic_config()
    data_loader = DefaultInitializingDataLoader(create_ones_mock_dataloader(nncf_config))
    nncf_config = register_default_init_args(nncf_config, train_loader=data_loader)
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/BasicBlock[0]/RELU_0', # 1
                                     'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]/batch_norm_0']
    orig_onnx_model_path = "resnet18.onnx"
    onnx_model_without_block_path = "resnet18_with_one_skipped_block.onnx"
 
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    # export model to onnx
    ctx = compressed_model._compressed_context
    compression_ctrl.export_model(orig_onnx_model_path)

    ctx._elastic_depth = True # activate mode with elastic depth
    ctx.set_active_skipped_block([0])
    compression_ctrl.export_model(onnx_model_without_block_path)

    # load onnx graphs
    onnx_resnet18_without_one_block = onnx.load_model(onnx_model_without_block_path)
    onnx_resnet18_orig = onnx.load_model(orig_onnx_model_path)

    #count of node in skipped block  == 5
    assert len(onnx_resnet18_orig.graph.node) == 69
    assert len(onnx_resnet18_without_one_block.graph.node) == 67

    input_tensor = np.ones(nncf_config['input_info']['sample_size'])
    torch_input = torch.tensor(input_tensor, dtype=torch.float32)
    with torch.no_grad():
        torch_model_output = compressed_model(torch_input)

    # ONNXRuntime
    sess = rt.InferenceSession(onnx_model_without_block_path)
    input_name = sess.get_inputs()[0].name
    onnx_model_output = sess.run(None, {input_name: input_tensor.astype(np.float32)})[0]
    assert np.allclose(torch_model_output.numpy(), onnx_model_output, rtol=1e-5, atol=1e-3)

def test_skip_one_block_resnet50(mocker):
    model = ResNet50()
    nncf_config = get_basic_config()
    data_loader = DefaultInitializingDataLoader(create_ones_mock_dataloader(nncf_config))
    nncf_config = register_default_init_args(nncf_config, train_loader=data_loader)
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/Bottleneck[1]/RELU_2', # 1
                                     'ResNet/Sequential[layer1]/Bottleneck[2]/RELU_2']
    _, compressed_model = create_compressed_model(model, nncf_config)

    ctx = compressed_model._compressed_context
    spy_agent = mocker.spy(NNCFConv2d, "__call__")
    ctx._elastic_depth = True # activate mode with elastic depth
    ctx.set_active_skipped_block([0])
    compressed_model(torch.ones([1, 3, 224, 224]))

    assert id(spy_agent.call_args_list[8][0][1]) == id(spy_agent.call_args_list[9][0][1])\
         == id(spy_agent.call_args_list[10][0][1]) # TracedTensor

    # check torch.tensor.data by element
    assert (spy_agent.call_args_list[8][0][1] == spy_agent.call_args_list[9][0][1]).sum() == np.prod(spy_agent.call_args_list[9][0][1].shape)

    spy_agent.reset_mock()

    ctx._elastic_depth = False 
    compressed_model(torch.ones([1, 3, 224, 224]))

    assert id(spy_agent.call_args_list[8][0][1]) != id(spy_agent.call_args_list[9][0][1]) # TracedTensor


def test_skip_diff_blocks_on_resnet50(mocker):
    model = ResNet50()
    nncf_config = get_basic_config()
    data_loader = DefaultInitializingDataLoader(create_ones_mock_dataloader(nncf_config))
    nncf_config = register_default_init_args(nncf_config, train_loader=data_loader)
    nncf_config['skipped_blocks'] = [['ResNet/Sequential[layer1]/Bottleneck[1]/RELU_2',
                                      'ResNet/Sequential[layer1]/Bottleneck[2]/RELU_2'],
                                     ['ResNet/Sequential[layer2]/Bottleneck[2]/RELU_2',
                                      'ResNet/Sequential[layer2]/Bottleneck[3]/RELU_2'],
                                     ['ResNet/Sequential[layer3]/Bottleneck[4]/RELU_2',
                                      'ResNet/Sequential[layer3]/Bottleneck[5]/RELU_2'],
                                     ['ResNet/Sequential[layer4]/Bottleneck[1]/RELU_2',
                                      'ResNet/Sequential[layer4]/Bottleneck[2]/RELU_2']
                                    ]
    _, compressed_model = create_compressed_model(model, nncf_config)

    ctx = compressed_model._compressed_context
    ctx._elastic_depth = True # activate mode with elastic depth
    ctx.set_active_skipped_block([0, 1, 2, 3])
    compressed_model(torch.ones([1, 3, 224, 224]))

    ctx.set_active_skipped_block([0, 1])
    compressed_model(torch.ones([1, 3, 224, 224]))
