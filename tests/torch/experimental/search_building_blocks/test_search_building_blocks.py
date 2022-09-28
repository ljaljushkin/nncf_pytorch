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
import json
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import os
import pytest
import torch
from functools import partial
from torchvision.models import MobileNetV2

from examples.torch.common.models import efficient_net
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlocks
from nncf.experimental.torch.search_building_blocks.search_blocks import GroupedBlockIDs
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from tests.common.helpers import TEST_ROOT
from tests.torch import test_models
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.test_elastic_depth import INCEPTION_INPUT_SIZE
from tests.torch.nas.test_elastic_depth import RESNET50_INPUT_SIZE
from tests.torch.test_models import ResNet18
from tests.torch.test_models import squeezenet1_0
from tests.torch.test_models.inceptionv3 import Inception3
from tests.torch.test_models.resnet import ResNet50

REF_BUILDING_BLOCKS_FOR_RESNET = [
    BuildingBlock('ResNet/Sequential[layer1]/Bottleneck[0]/relu_2',
                  'ResNet/Sequential[layer1]/Bottleneck[1]/relu_2'),

    BuildingBlock('ResNet/Sequential[layer1]/Bottleneck[1]/relu_2',
                  'ResNet/Sequential[layer1]/Bottleneck[2]/relu_2'),

    BuildingBlock('ResNet/Sequential[layer2]/Bottleneck[0]/relu_2',
                  'ResNet/Sequential[layer2]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer2]/Bottleneck[1]/relu_2',
                  'ResNet/Sequential[layer2]/Bottleneck[2]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer2]/Bottleneck[2]/relu_2',
                  'ResNet/Sequential[layer2]/Bottleneck[3]/relu_2'),

    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[0]/relu_2',
                  'ResNet/Sequential[layer3]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[1]/relu_2',
                  'ResNet/Sequential[layer3]/Bottleneck[2]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[2]/relu_2',
                  'ResNet/Sequential[layer3]/Bottleneck[3]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[3]/relu_2',
                  'ResNet/Sequential[layer3]/Bottleneck[4]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[4]/relu_2',
                  'ResNet/Sequential[layer3]/Bottleneck[5]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer4]/Bottleneck[0]/relu_2',
                  'ResNet/Sequential[layer4]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer4]/Bottleneck[1]/relu_2',
                  'ResNet/Sequential[layer4]/Bottleneck[2]/relu_2')
]

REF_BUILDING_BLOCKS_FOR_MOBILENETV2 = [
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[3]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[6]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[8]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[8]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[9]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[9]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[10]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[12]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[12]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[13]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[15]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[15]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[16]/__add___0')
]

REF_BUILDING_BLOCKS_FOR_SQUEEZENET = [
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[3]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/relu__0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[4]/cat_0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[7]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[7]/cat_0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[8]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/relu__0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[9]/cat_0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[12]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[12]/cat_0')
]

REF_BIG_BUILDING_BLOCKS_FOR_INCEPTION_V3 = [
    BuildingBlock('Inception3/InceptionA[Mixed_5c]/cat_0',
                  'Inception3/InceptionA[Mixed_5d]/cat_0'),
    BuildingBlock('Inception3/InceptionB[Mixed_6a]/cat_0',
                  'Inception3/InceptionC[Mixed_6b]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6b]/cat_0',
                  'Inception3/InceptionC[Mixed_6c]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6c]/cat_0',
                  'Inception3/InceptionC[Mixed_6d]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6d]/cat_0',
                  'Inception3/InceptionC[Mixed_6e]/cat_0'),
    BuildingBlock('Inception3/InceptionE[Mixed_7b]/cat_2',
                  'Inception3/InceptionE[Mixed_7c]/cat_2')
]

REF_SMALL_BUILDING_BLOCKS_FOR_INCEPTION_V3 = [
    BuildingBlock('Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_3]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_3]/relu__0',
                  'Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_3]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_3]/relu__0',
                  'Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_3]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_3]/relu__0',
                  'Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7_3]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_3]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_3]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_4]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_5]/relu__0'),
    BuildingBlock('Inception3/InceptionD[Mixed_7a]/BasicConv2d[branch7x7x3_2]/relu__0',
                  'Inception3/InceptionD[Mixed_7a]/BasicConv2d[branch7x7x3_3]/relu__0')
]

REF_BUILDING_BLOCKS_FOR_ResNext = [
    BuildingBlock('ResNeXt/Sequential[layer1]/Block[0]/relu_2', 'ResNeXt/Sequential[layer1]/Block[1]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer1]/Block[1]/relu_2', 'ResNeXt/Sequential[layer1]/Block[2]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer2]/Block[0]/relu_2', 'ResNeXt/Sequential[layer2]/Block[1]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer2]/Block[1]/relu_2', 'ResNeXt/Sequential[layer2]/Block[2]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer3]/Block[0]/relu_2', 'ResNeXt/Sequential[layer3]/Block[1]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer3]/Block[1]/relu_2', 'ResNeXt/Sequential[layer3]/Block[2]/relu_2')]

REF_BUILDING_BLOCKS_FOR_PNASNetB = [
    BuildingBlock('PNASNet/relu_0', 'PNASNet/Sequential[layer1]/CellB[0]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[0]/relu_2', 'PNASNet/Sequential[layer1]/CellB[1]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[1]/relu_2', 'PNASNet/Sequential[layer1]/CellB[2]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[2]/relu_2', 'PNASNet/Sequential[layer1]/CellB[3]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[3]/relu_2', 'PNASNet/Sequential[layer1]/CellB[4]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[4]/relu_2', 'PNASNet/Sequential[layer1]/CellB[5]/relu_2'),
    BuildingBlock('PNASNet/CellB[layer2]/relu_2', 'PNASNet/Sequential[layer3]/CellB[0]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[0]/relu_2', 'PNASNet/Sequential[layer3]/CellB[1]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[1]/relu_2', 'PNASNet/Sequential[layer3]/CellB[2]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[2]/relu_2', 'PNASNet/Sequential[layer3]/CellB[3]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[3]/relu_2', 'PNASNet/Sequential[layer3]/CellB[4]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[4]/relu_2', 'PNASNet/Sequential[layer3]/CellB[5]/relu_2'),
    BuildingBlock('PNASNet/CellB[layer4]/relu_2', 'PNASNet/Sequential[layer5]/CellB[0]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[0]/relu_2', 'PNASNet/Sequential[layer5]/CellB[1]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[1]/relu_2', 'PNASNet/Sequential[layer5]/CellB[2]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[2]/relu_2', 'PNASNet/Sequential[layer5]/CellB[3]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[3]/relu_2', 'PNASNet/Sequential[layer5]/CellB[4]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[4]/relu_2', 'PNASNet/Sequential[layer5]/CellB[5]/relu_2')]

REF_BUILDING_BLOCKS_FOR_EFFICIENT_NET = [
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[0]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[0]/MemoryEfficientSwish[_swish]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[0]/MemoryEfficientSwish[_swish]/__mul___0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[0]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[1]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[1]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[3]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[3]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[5]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[5]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[11]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[11]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/__mul___0'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[15]/NNCFUserConv2dStaticSamePadding[_depthwise_conv]/ZeroPad2d[static_padding]/pad_0',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[15]/MemoryEfficientSwish[_swish]/__mul___1'),
    BuildingBlock(
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[15]/MemoryEfficientSwish[_swish]/__mul___1',
        'EfficientNet/ModuleList[_blocks]/MBConvBlock[15]/__mul___0')]

REF_GROUP_DEPENDENT_RESNET50 = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8, 9], 3: [10, 11]}
REF_GROUP_DEPENDENT_MOBILENETV2 = {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [6, 7], 4: [8, 9]}
REF_BIG_GROUP_DEPENDENT_INCEPTIONV3 = {0: [0], 1: [1, 2, 3, 4], 2: [5]}
REF_SMALL_GROUP_DEPENDENT_INCEPTIONV3 = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6], 4: [7, 8, 9], 5: [10]}
REF_GROUP_DEPENDENT_SQUEEZNET = {0: [0, 1], 1: [2], 2: [3, 4], 3: [5]}
REF_GROUP_DEPENDENT_PNASNETB = {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11], 2: [12, 13, 14, 15, 16, 17]}
REF_GROUP_DEPENDENT_RESNEXT = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
REF_GROUP_DEPENDENT_SSD_MOBILENET = {0: [0], 1: [1], 2: [2, 3, 4, 5, 6, 7, 8, 9, 10], 3: [11]}
REF_GROUP_DEPENDENT_EFFICIENT_NET = {0: [0, 1], 1: [2], 2: [3, 4], 3: [5], 4: [6, 7], 5: [8], 6: [9, 10], 7: [11, 12],
                                     8: [13, 14], 9: [15, 16], 10: [17, 18], 11: [19], 12: [20, 21], 13: [22, 23],
                                     14: [24, 25], 15: [26, 27]}


def check_blocks_and_groups(name, actual_blocks: BuildingBlocks, actual_group_dependent: GroupedBlockIDs):
    ref_file_dir = TEST_ROOT.joinpath('torch', 'data', 'search_building_block')
    ref_file_path = ref_file_dir.joinpath(name)
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(ref_file_dir):
            os.makedirs(ref_file_dir)
        with ref_file_path.open('w', encoding='utf8') as f:
            actual_state = {
                'blocks': [block.get_state() for block in actual_blocks],
                'group_dependent': actual_group_dependent
            }
            json.dump(actual_state, f, indent=4)

    with ref_file_path.open('r') as f:
        ref_state = json.load(f)
        ref_blocks = [BuildingBlock.from_state(state) for state in ref_state['blocks']]
        ref_group_dependent = {int(k): v for k, v in ref_state['group_dependent'].items()}
        assert ref_blocks == actual_blocks
        assert ref_group_dependent == actual_group_dependent


class BuildingBlockParamsCase:
    def __init__(self,
                 model_creator: Union[Type[torch.nn.Module], Callable[[], torch.nn.Module]],
                 input_sizes: List[int],
                 min_block_size: int = 5,
                 max_block_size: int = 50,
                 name: Optional[str] = None,
                 hw_fused_ops: bool = True):
        self.model_creator = model_creator
        self.input_sizes = input_sizes
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.name = name
        self.hw_fused_ops = hw_fused_ops

    def __str__(self):
        name = self.name
        if not name and hasattr(self.model_creator, '__name__'):
            name = self.model_creator.__name__
        assert name, 'Can\'t define name from the model (usually due to partial), please specify it explicitly'
        return name


LIST_BB_PARAMS_CASES = [
    BuildingBlockParamsCase(ResNet50, RESNET50_INPUT_SIZE, hw_fused_ops=False),
    BuildingBlockParamsCase(MobileNetV2, RESNET50_INPUT_SIZE),
    BuildingBlockParamsCase(Inception3, INCEPTION_INPUT_SIZE,
                            min_block_size=23, name='Inception3_big_blocks'),
    BuildingBlockParamsCase(Inception3, INCEPTION_INPUT_SIZE,
                            min_block_size=4, max_block_size=5, name='Inception3_small_blocks'),
    BuildingBlockParamsCase(squeezenet1_0, RESNET50_INPUT_SIZE),
    BuildingBlockParamsCase(test_models.ResNeXt29_32x4d, [1, 3, 32, 32], hw_fused_ops=False),
    BuildingBlockParamsCase(test_models.PNASNetB, [1, 3, 32, 32]),
    BuildingBlockParamsCase(test_models.ssd_mobilenet, [2, 3, 300, 300],
                            min_block_size=2, max_block_size=7),
    BuildingBlockParamsCase(partial(efficient_net, model_name='efficientnet-b0'), [10, 3, 240, 240],
                            name='efficientnet-b0', min_block_size=2, max_block_size=7)
]


@pytest.mark.parametrize('desc', LIST_BB_PARAMS_CASES, ids=map(str, LIST_BB_PARAMS_CASES))
def test_building_block(desc: BuildingBlockParamsCase):
    model = desc.model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=desc.input_sizes)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ext_blocks, group_dependent = get_building_blocks(compressed_model,
                                                      max_block_size=desc.max_block_size,
                                                      min_block_size=desc.min_block_size,
                                                      hw_fused_ops=desc.hw_fused_ops)
    skipped_blocks = [eb.basic_block for eb in ext_blocks]
    check_blocks_and_groups(str(desc), skipped_blocks, group_dependent)


class SearchBBlockAlgoParamsCase:
    def __init__(self,
                 min_block_size: int = 1,
                 max_block_size: int = 100,
                 hw_fused_ops: bool = False,
                 ref_blocks=None):
        self.max_block_size = max_block_size
        self.min_block_size = min_block_size
        self.ref_blocks = [] if ref_blocks is None else ref_blocks
        self.hw_fused_ops = hw_fused_ops


@pytest.mark.parametrize('algo_params', (
        (
                SearchBBlockAlgoParamsCase(max_block_size=5,
                                           ref_blocks=[]),
                SearchBBlockAlgoParamsCase(max_block_size=6,
                                           ref_blocks=[
                                               BuildingBlock('ResNet/MaxPool2d[maxpool]/max_pool2d_0',
                                                             'ResNet/Sequential[layer1]/BasicBlock[0]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer1]/BasicBlock[0]/relu_1',
                                                             'ResNet/Sequential[layer1]/BasicBlock[1]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer2]/BasicBlock[0]/relu_1',
                                                             'ResNet/Sequential[layer2]/BasicBlock[1]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer3]/BasicBlock[0]/relu_1',
                                                             'ResNet/Sequential[layer3]/BasicBlock[1]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer4]/BasicBlock[0]/relu_1',
                                                             'ResNet/Sequential[layer4]/BasicBlock[1]/relu_1')
                                           ]),
                SearchBBlockAlgoParamsCase(min_block_size=8,
                                           max_block_size=14,
                                           ref_blocks=[
                                               BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                             "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1")
                                           ]),
                SearchBBlockAlgoParamsCase(min_block_size=7,
                                           max_block_size=7,
                                           ref_blocks=[
                                               BuildingBlock('ResNet/Sequential[layer1]/BasicBlock[0]/__iadd___0',
                                                             'ResNet/Sequential[layer1]/BasicBlock[1]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer2]/BasicBlock[0]/__iadd___0',
                                                             'ResNet/Sequential[layer2]/BasicBlock[1]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer3]/BasicBlock[0]/__iadd___0',
                                                             'ResNet/Sequential[layer3]/BasicBlock[1]/relu_1'),
                                               BuildingBlock('ResNet/Sequential[layer4]/BasicBlock[0]/__iadd___0',
                                                             'ResNet/Sequential[layer4]/BasicBlock[1]/relu_1')
                                           ]))
))
def test_building_block_algo_param(algo_params: SearchBBlockAlgoParamsCase):
    model = ResNet18()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ext_blocks, _ = get_building_blocks(compressed_model,
                                        min_block_size=algo_params.min_block_size,
                                        max_block_size=algo_params.max_block_size,
                                        )
    blocks = [eb.basic_block for eb in ext_blocks]
    assert blocks == algo_params.ref_blocks
