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
from typing import Union

import pytest
from functools import partial
from torch import nn
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForQuestionAnswering

from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks_info
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
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]/dropout_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 1
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 2
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 3
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 4
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 5
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 6
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 7
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 8
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 9
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 10
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertOutput[output]/__add___0',
        28
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        6
    ),

    # 11
    RefBlockInfo(
        BuildingBlockType.Unknown,
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]/layer_norm_0',
        'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/ModuleList[layer]/BertLayer[11]/BertOutput[output]/__add___0',
        28
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[0]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[0]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[1]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[1]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[2]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[2]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[3]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[3]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[4]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[4]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[5]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[5]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[6]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[6]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[7]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[7]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/ViTOutput[output]/__add___0',
        6
    ),

    # 8
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[8]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/ViTOutput[output]/__add___0',
        6
    ),

    # 9
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[9]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/ViTOutput[output]/__add___0',
        6
    ),

    # 10
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[10]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
    ),
    RefBlockInfo(
        BuildingBlockType.FF,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[11]/__add___0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[11]/ViTOutput[output]/__add___0',
        6
    ),

    # 11
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[11]/LayerNorm[layernorm_before]/layer_norm_0',
        'ViTForImageClassification/ViTModel[vit]/ViTEncoder[encoder]/ModuleList[layer]/ViTLayer[11]/ViTAttention[attention]/ViTSelfOutput[output]/NNCFLinear[dense]/linear_0',
        19
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
        BuildingBlockType.Unknown,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2FeatureProjection[feature_projection]/Dropout[dropout]/dropout_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/LayerNorm[layer_norm]/layer_norm_0',
        7
    ),
    RefBlockInfo(
        BuildingBlockType.MSHA,
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/Dropout[dropout]/dropout_0',
        'Wav2Vec2ForSequenceClassification/Wav2Vec2Model[wav2vec2]/Wav2Vec2Encoder[encoder]/ModuleList[layers]/Wav2Vec2EncoderLayer[0]/__add___0',
        27
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
    )
]


@pytest.fixture(name='desc', scope='function', params=LIST_CASES, ids=map(lambda x: x.name, LIST_CASES))
def fixture_transformer_search_params_desc(request):
    return request.param


def test_transformer_building_blocks(desc: TransformerSearchBBlockParamsCase):
    model = desc.model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_info=desc.input_info)
    nncf_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    blocks, _, group_dependent = get_building_blocks(nncf_model, allow_nested_blocks=False)
    blocks_info = get_building_blocks_info(blocks, nncf_model)

    assert len(blocks_info) == len(desc.ref_blocks), 'different number of blocks'
    for act_bi, ref_bi in zip(blocks_info, desc.ref_blocks):
        assert act_bi.block_type == ref_bi.block_type
        assert act_bi.building_block.start_node_name == ref_bi.start_node_name
        assert act_bi.building_block.end_node_name == ref_bi.end_node_name
        assert len(act_bi.op_addresses) == ref_bi.num_ops
