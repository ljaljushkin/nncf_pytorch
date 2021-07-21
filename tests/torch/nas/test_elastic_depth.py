import torch
import numpy as np
import onnx
import onnxruntime as rt
from torch import nn
from typing import List

from nncf.config import NNCFConfig
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.initialization import DefaultInitializingDataLoader
from nncf.torch.layers import NNCFConv2d
from nncf.torch.dynamic_graph.context import get_current_context
from tests.torch.test_models.resnet import ElasticResNet, ResNet18, ResNet50, ResNet50__elastic
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import create_conv

RESNET50_INPUT_SIZE = [1, 3, 224, 224]

def get_basic_config(sample_size=[1, 3, 224, 224]):
    config = NNCFConfig()
    config.update(dict({
        "model": "basic_quant_conv",
        "input_info":
            {
                "sample_size": sample_size,
            }
    }))
    return config

def test_skip_one_block_resnet18(mocker):
    model = ResNet18()
    nncf_config = get_basic_config()
    data_loader = DefaultInitializingDataLoader(create_ones_mock_dataloader(nncf_config))
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/BasicBlock[0]/relu_0', # 1
                                     'ResNet/Sequential[layer1]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0']
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
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/BasicBlock[0]/relu_0', # 1
                                     'ResNet/Sequential[layer1]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0']
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
    nncf_config['skipped_blocks'] = ['ResNet/Sequential[layer1]/Bottleneck[1]/relu_2', # 1
                                     'ResNet/Sequential[layer1]/Bottleneck[2]/relu_2']
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
    nncf_config['skipped_blocks'] = [['ResNet/Sequential[layer1]/Bottleneck[1]/relu_2',
                                      'ResNet/Sequential[layer1]/Bottleneck[2]/relu_2'],
                                     ['ResNet/Sequential[layer2]/Bottleneck[2]/relu_2',
                                      'ResNet/Sequential[layer2]/Bottleneck[3]/relu_2'],
                                     ['ResNet/Sequential[layer3]/Bottleneck[4]/relu_2',
                                      'ResNet/Sequential[layer3]/Bottleneck[5]/relu_2'],
                                     ['ResNet/Sequential[layer4]/Bottleneck[1]/relu_2',
                                      'ResNet/Sequential[layer4]/Bottleneck[2]/relu_2']
                                    ]
    _, compressed_model = create_compressed_model(model, nncf_config)

    ctx = compressed_model._compressed_context
    ctx._elastic_depth = True # activate mode with elastic depth
    ctx.set_active_skipped_block([0, 1, 2, 3])
    compressed_model(torch.ones([1, 3, 224, 224]))

    ctx.set_active_skipped_block([0, 1])
    compressed_model(torch.ones([1, 3, 224, 224]))

count = 0
def init_weight(m):
    global count

    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        count += 1
        m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.fill_(0.01)

def reset_bn_stats(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()

class BasicConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self, depth=3):
        super().__init__()
        self._depth = depth
        self._skipped_layers = []
        self.conv1 = create_conv(1, 3, 2, weight_init=1, bias_init=1)
        self.branch_with_blocks = nn.Sequential()
        for id in range(depth):
            conv = create_conv(3, 3, 1, weight_init=id+1, bias_init=id+1)
            self.branch_with_blocks.add_module('conv{}'.format(id), conv)

    def forward(self, x):
        output = self.conv1(x)
        for name, module in dict(self.branch_with_blocks._modules).items():
            if name not in self._skipped_layers:
                output = module(output)
        return output

    def set_skipped_layers(self, skipped_layers: List = []):
        self._skipped_layers = skipped_layers

def get_ref_output_model_after_backward__with_manual_skipping():
    # forward and backward with "manual" mechanism skipping block

    model = BasicConvTestModel(depth=3)
    optimizer_for_model = torch.optim.Adam(model.parameters(), lr=0.01)

    # set skipped layer
    model.set_skipped_layers(['conv1'])

    output = model(torch.ones(model.INPUT_SIZE))
    optimizer_for_model.zero_grad()
    output.sum().backward()
    optimizer_for_model.step()

    output = model(torch.ones(model.INPUT_SIZE))

    return output

def get_model_with_elastic_depth(model, input_size, skipped_blocks):
    nncf_config = get_basic_config(input_size)
    nncf_config['skipped_blocks'] = skipped_blocks

    _, compressed_model = create_compressed_model(model, nncf_config)

    return compressed_model

def get_ref_output_model_after_backward__with_elastic_depth():
    # forward and backward with elastic_depth
    target_model = BasicConvTestModel(depth=3)
    skipped_block = ['BasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0',
                     'BasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0']
    compressed_model = get_model_with_elastic_depth(target_model, target_model.INPUT_SIZE, skipped_block)

    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=0.01)

    # pylint:disable=protected-access
    ctx = compressed_model._compressed_context
    ctx._elastic_depth = True
    ctx.set_active_skipped_block([0])

    output = compressed_model(torch.ones(target_model.INPUT_SIZE))
    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()

    output = compressed_model(torch.ones(target_model.INPUT_SIZE))

    return output

def get_ref_output_resnet50_after_backward__with_elastic_depth():
    # forward and backward with elastic_depth
    target_model = ResNet50()
    with torch.no_grad():
        target_model.apply(init_weight)
    skipped_block = ['ResNet/Sequential[layer2]/Bottleneck[0]/relu_2',
                     'ResNet/Sequential[layer2]/Bottleneck[1]/relu_2']

    compressed_model = get_model_with_elastic_depth(target_model, RESNET50_INPUT_SIZE,  skipped_block)

    with torch.no_grad():
        compressed_model.apply(reset_bn_stats)
    compressed_model.train()
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=0.01)

    # pylint:disable=protected-access
    ctx = compressed_model._compressed_context
    ctx._elastic_depth = True
    ctx.set_active_skipped_block([0])

    output = compressed_model(torch.ones(RESNET50_INPUT_SIZE))
    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()

    output = compressed_model(torch.ones(RESNET50_INPUT_SIZE))

    return output


def get_output_model__with_manual_skipping():
    model = BasicConvTestModel(depth=3)
    output = model(torch.ones(model.INPUT_SIZE))
    return output

def get_output_model__with_elastic_depth():
    target_model = BasicConvTestModel(depth=3)
    skipped_block = ['BasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0',
                     'BasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0']
    compressed_model = get_model_with_elastic_depth(target_model, target_model.INPUT_SIZE, skipped_block)
    output = compressed_model(torch.ones(target_model.INPUT_SIZE))

    return output

def get_ref_output_resnet50_after_backward__with_manual_skipping():
    # forward and backward with "manual" mechanism skipping block
    idx_skip_blocks = [4]
    model = ResNet50__elastic(idx_skip_blocks)
    with torch.no_grad():
        model.apply(init_weight)
    optimizer_for_model = torch.optim.Adam(model.parameters(), lr=0.01)

    with torch.no_grad():
        model.apply(reset_bn_stats)
    model.train()
    output = model(torch.ones(RESNET50_INPUT_SIZE))
    optimizer_for_model.zero_grad()
    output.sum().backward()
    optimizer_for_model.step()
    output = model(torch.ones(RESNET50_INPUT_SIZE))

    return output

def test_correct_grad_when_block_skipped():
    output_ref = get_ref_output_model_after_backward__with_manual_skipping()
    output = get_ref_output_model_after_backward__with_elastic_depth()

    assert (output_ref == output).sum() == np.prod(output.shape)

def test_correct_output_with_active_skipped_block():
    output_ref = get_output_model__with_manual_skipping()
    output = get_output_model__with_elastic_depth()
    assert (output_ref == output).sum() == np.prod(output.shape)

def test_check_dinamic_graph_not_grow():
    model = ResNet50()
    nncf_config = get_basic_config()
    nncf_config['skipped_blocks'] = [['ResNet/Sequential[layer1]/Bottleneck[1]/relu_2', # 3
                                      'ResNet/Sequential[layer1]/Bottleneck[2]/relu_2'],
                                     ['ResNet/Sequential[layer2]/Bottleneck[2]/relu_2', # 7
                                      'ResNet/Sequential[layer2]/Bottleneck[3]/relu_2'],
                                     ['ResNet/Sequential[layer3]/Bottleneck[4]/relu_2', # 13
                                      'ResNet/Sequential[layer3]/Bottleneck[5]/relu_2'],
                                     ['ResNet/Sequential[layer4]/Bottleneck[1]/relu_2', # 16
                                      'ResNet/Sequential[layer4]/Bottleneck[2]/relu_2']
                                    ]
    _, compressed_model = create_compressed_model(model, nncf_config)

    #pylint: disable=protected-access
    ctx = compressed_model._compressed_context
    nodes_count = ctx.graph.get_nodes_count()
    ctx._elastic_depth = True # activate mode with elastic depth
    ctx.set_active_skipped_block([0, 1, 2, 3])

    for _ in range(10):
        compressed_model(torch.ones([1, 3, 224, 224]))
        assert nodes_count == ctx.graph.get_nodes_count()

def test_correct_grad_when_block_skipped__resnet50():
    from nncf.torch.utils import manual_seed
    import torch.backends.cudnn as cudnn

    manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False

    output_ref = get_ref_output_resnet50_after_backward__with_manual_skipping()
    output = get_ref_output_resnet50_after_backward__with_elastic_depth()

    assert (output_ref == output).sum() == np.prod(output.shape)
