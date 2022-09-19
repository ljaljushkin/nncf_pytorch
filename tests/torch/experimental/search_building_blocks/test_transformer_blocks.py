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
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import numpy as np
import pytest
import torch
from functools import partial
from torch import nn
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForQuestionAnswering

from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks_info
from nncf.experimental.torch.search_building_blocks.search_blocks import get_indexes_of_overlapping_blocks
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available


class RefBlockInfo:
    def __init__(self,
                 block_type: BuildingBlockType,
                 start_node_name: str,
                 end_node_name: str,
                 num_ops: int):
        self.block_type = block_type
        self.start_node_name = start_node_name
        self.end_node_name = end_node_name
        self.num_ops = num_ops


class TransformerSearchBBlockParamsCase:
    def __init__(self,
                 name: str,
                 input_info: Union[List, Dict],
                 model_creator: Callable[[], nn.Module],
                 ref_blocks: List[RefBlockInfo]):
        self.input_info = input_info
        self.model_creator = model_creator
        self.name = name
        self.ref_blocks = ref_blocks


BERT_REF_BLOCKS = [
    # 0
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]/dropout_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 1
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 2
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 3
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 4
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 5
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 6
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 7
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 8
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 9
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 10
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 11
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        23
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[11]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),
]

VIT_REF_BLOCKS = [
    # 0
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEmbeddings[embeddings]/Dropout[dropout]/dropout_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[0]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[0]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[0]/ViTOutput[output]/__add___0',
        6
    ),

    # 1
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[0]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[1]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[1]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[1]/ViTOutput[output]/__add___0',
        6
    ),

    # 2
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[1]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[2]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[2]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[2]/ViTOutput[output]/__add___0',
        6
    ),

    # 3
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[2]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[3]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[3]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[3]/ViTOutput[output]/__add___0',
        6
    ),

    # 4
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[3]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[4]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[4]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[4]/ViTOutput[output]/__add___0',
        6
    ),

    # 5
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[4]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[5]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[5]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[5]/ViTOutput[output]/__add___0',
        6
    ),

    # 6
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[5]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[6]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[6]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[6]/ViTOutput[output]/__add___0',
        6
    ),

    # 7
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[6]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[7]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[7]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[7]/ViTOutput[output]/__add___0',
        6
    ),

    # 8
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[7]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/ViTOutput[output]/__add___0',
        6
    ),

    # 9
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/ViTOutput[output]/__add___0',
        6
    ),

    # 10
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/ViTOutput[output]/__add___0',
        6
    ),

    # 11
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/ViTOutput[output]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[11]/__add___0',
        22
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[11]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/LayerNorm[layernorm]/layer_norm_0',
        7
    ),
]

WAVE2VEC2_REF_BLOCKS = [
    # 0
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2FeatureProjection[feature_projection]/Dropout[dropout]/dropout_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[0]/LayerNorm[layer_norm]/layer_norm_0',
        36
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[0]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[0]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 1
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[0]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[1]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[1]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[1]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 2
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[1]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[2]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[2]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[2]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 3
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[2]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[3]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[3]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[3]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 4
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[3]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[4]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[4]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[4]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 5
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[4]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[5]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[5]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[5]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 6
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[5]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[6]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[6]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[6]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 7
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[6]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[7]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[7]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[7]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 8
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[7]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[8]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[8]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[8]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 9
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[8]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[9]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[9]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[9]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 10
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[9]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[10]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[10]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[10]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),

    # 11
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[10]/LayerNorm[final_layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[11]/LayerNorm[layer_norm]/layer_norm_0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[11]/LayerNorm[layer_norm]/layer_norm_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[11]/LayerNorm[final_layer_norm]/layer_norm_0',
        7
    ),
]


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.o = nn.Linear(768, 768)
        self.sm = nn.Softmax()

    def forward(self, x):
        # assert x.shape == [384, 768]
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        k = k.view(-1, 12, 64).permute(1, 0, 2)
        q = q.view(-1, 12, 64).permute(1, 2, 0)
        v = v.view(-1, 12, 64).permute(1, 0, 2)
        x = self.sm(torch.matmul(k, q)) / np.sqrt(1 / 384)
        x = torch.matmul(x, v)
        x = x.permute(1, 0, 2).contiguous().view(-1, 768)
        return self.o(x)


LIST_CASES = [
    TransformerSearchBBlockParamsCase(
        name='BERT',
        input_info=[dict(sample_size=[1, 10], type='long')] * 3,
        model_creator=partial(AutoModelForQuestionAnswering.from_pretrained,
                              'bert-base-uncased'),
        ref_blocks=BERT_REF_BLOCKS,
    ),
    TransformerSearchBBlockParamsCase(
        name='ViT',
        input_info=dict(sample_size=[1, 3, 224, 224]),
        model_creator=partial(AutoModelForImageClassification.from_pretrained,
                              'google/vit-base-patch16-224'),
        ref_blocks=VIT_REF_BLOCKS,
    ),
    TransformerSearchBBlockParamsCase(
        name='wave2vec 2.0',
        input_info=dict(sample_size=[1, 400]),
        model_creator=partial(AutoModelForAudioClassification.from_pretrained,
                              'anton-l/wav2vec2-base-ft-keyword-spotting'),
        ref_blocks=WAVE2VEC2_REF_BLOCKS,
    ),
    TransformerSearchBBlockParamsCase(
        name='one MSHA',
        input_info=dict(sample_size=[384, 768]),
        model_creator=SelfAttention,
        ref_blocks=[
            RefBlockInfo(BuildingBlockType.MSHA, '/nncf_model_input_0', 'SelfAttention/NNCFLinear[o]/linear_0', 17)
        ]
    )
]


@pytest.fixture(name='desc', scope='function', params=LIST_CASES, ids=map(lambda x: x.name, LIST_CASES))
def fixture_transformer_search_params_desc(request):
    return request.param


# TODO: debug on a single synthetic MSHA model
def test_transformer_building_blocks(desc: TransformerSearchBBlockParamsCase):
    model = desc.model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_info=desc.input_info)
    nncf_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ext_blocks, group_dependent = get_building_blocks(nncf_model)
    blocks_info = get_building_blocks_info(ext_blocks, nncf_model)

    assert len(blocks_info) == len(desc.ref_blocks), 'different number of blocks'
    print(len(blocks_info))
    for act_bi, ref_bi in zip(blocks_info, desc.ref_blocks):
        print(act_bi.block_type)
        print(len(act_bi.op_addresses))
        print(f'{act_bi.building_block.start_node_name} ------ {act_bi.building_block.end_node_name}')
        # print(*act_bi.op_addresses, sep='\n')
        print('\n\n')
        assert act_bi.block_type == ref_bi.block_type
        assert act_bi.building_block.start_node_name == ref_bi.start_node_name
        assert act_bi.building_block.end_node_name == ref_bi.end_node_name
        assert len(act_bi.op_addresses) == ref_bi.num_ops


class FilterBlockTestDesc:
    def __init__(self,
                 start_ids: List[int],
                 end_ids: List[int],
                 overlapping_blocks_ids: Optional[Set[int]] = None,
                 num_ops_in_block: Optional[List[int]] = None,
                 name: Optional[str] = None):
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.overlapping_blocks_ids = overlapping_blocks_ids
        if self.overlapping_blocks_ids is None:
            self.overlapping_blocks_ids = set()
        self.num_ops_in_block = num_ops_in_block
        if self.num_ops_in_block is None:
            self.num_ops_in_block = [e - s for s, e in zip(self.start_ids, self.end_ids)]
        self.name = name
        if self.name is None:
            self.name = '__'.join(f'{s}:{e}' for s, e in zip(self.start_ids, self.end_ids))

    def __str__(self):
        return self.name


LIST_FILTER_BLOCK_DESCS = [
    FilterBlockTestDesc(
        name='empty',
        start_ids=[],
        end_ids=[],
    ),
    FilterBlockTestDesc(
        start_ids=[1],
        end_ids=[2],
    ),
    FilterBlockTestDesc(
        start_ids=[1, 2, 3, 4],
        end_ids=[2, 3, 4, 5],
    ),
    FilterBlockTestDesc(
        start_ids=[1, 2, 3, 4],
        end_ids=[5, 5, 5, 5],
        overlapping_blocks_ids={1, 2, 3}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 1, 1, 1],
        end_ids=[2, 3, 4, 5],
        overlapping_blocks_ids={0, 1, 2}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 1, 2, 2],
        end_ids=[2, 3, 3, 4],
        overlapping_blocks_ids={0, 2, 3}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 1, 2, 2],
        end_ids=[4, 3, 3, 4],
        overlapping_blocks_ids={1, 2, 3}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 2, 2, 1],
        end_ids=[4, 3, 4, 3],
        overlapping_blocks_ids={1, 2, 3}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 3, 3, 4, 5, 10, 11],
        end_ids=[4, 5, 6, 7, 6, 14, 12],
        overlapping_blocks_ids={1, 2, 4, 6}
    ),
    FilterBlockTestDesc(
        start_ids=[3, 10, 3, 5, 11, 1, 4],
        end_ids=[6, 14, 5, 6, 12, 4, 7],
        overlapping_blocks_ids={0, 2, 3, 4}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 2, 3, 4],
        end_ids=[5, 4, 6, 9],
        overlapping_blocks_ids={0, 1, 2}
    ),
    FilterBlockTestDesc(
        start_ids=[1, 3, 2, 4],
        end_ids=[5, 6, 4, 9],
        overlapping_blocks_ids={0, 1, 2}
    ),
    FilterBlockTestDesc(
        name='non_standard_num_ops',
        start_ids=[1, 2, 2, 1],
        end_ids=[4, 3, 4, 3],
        num_ops_in_block=[1, 10, 2, 11],
        overlapping_blocks_ids={0, 1, 2}
    ),
]


@pytest.fixture(name='filter_blocks_desc', scope='function', params=LIST_FILTER_BLOCK_DESCS,
                ids=map(str, LIST_FILTER_BLOCK_DESCS))
def fixture_filter_blocks_desc(request) -> FilterBlockTestDesc:
    return request.param


def test_filtering(filter_blocks_desc: FilterBlockTestDesc):
    actual_indexes_of_overlapping_blocks = get_indexes_of_overlapping_blocks(
        filter_blocks_desc.start_ids,
        filter_blocks_desc.end_ids,
        filter_blocks_desc.num_ops_in_block
    )
    assert actual_indexes_of_overlapping_blocks == filter_blocks_desc.overlapping_blocks_ids
