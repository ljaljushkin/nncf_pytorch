from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import pytest
import torch

# import nncf.torch  # TODO: why "import nncf" is not enough for calling patch_torch_operators??
from nncf.torch import patch_torch_operators
patch_torch_operators()  # more reliable, since it's not removed as unused
# TODO: why this import breaks BERT tests???
#             assert len(output_tensors_shapes) == 1 or len(set(output_tensors_shapes)) <= 1, node.node_name
#  >           output_tensors_shape = output_tensors_shapes[0]
#  E           IndexError: list index out of range
#  need to dump graph, seems like some operation is not wrapped because of order of imports?
#  ANSWER: gelu is not patched, graph is disjointed and linear has no output shapes, which is not expected.

from transformers import AutoModelForImageClassification
from transformers import RobertaConfig
from transformers import SwinConfig
from transformers import ViTConfig


from torch import nn
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import Wav2Vec2Config

from nncf import NNCFConfig
from nncf.experimental.common.pruning.nodes_grouping import MinimalDimensionBlock
from nncf.experimental.common.pruning.nodes_grouping import PruningNodeGroup
from nncf.experimental.common.pruning.nodes_grouping import get_pruning_groups
from nncf.experimental.torch.pruning.operations import PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES
from nncf.torch.layers import NNCF_PRUNING_MODULES_DICT
from nncf.torch.model_creation import create_nncf_network
from tests.torch.test_compressed_graph import GeneralModelDesc
from tests.torch.test_compressed_graph import IModelDesc




class SelfAttention(nn.Module):
    INPUT_SAMPLE_SIZES = ([384, 768])
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.o = nn.Linear(768, 768)
        self.sm = nn.Softmax()

    def forward(self, x):
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


class DiffNumBranchesOnJoining(nn.Module):
    INPUT_SAMPLE_SIZES = ([1, 3])

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(1, 6)
        self.k = nn.Linear(1, 6)
        self.o = nn.Linear(2, 4)

    def forward(self, x):
        k = self.k(x)  # [6]
        q = self.q(x)  # [6]
        k = k.view(2, 3).permute(1, 0)  # [3, 2]
        q = q.view(2, 3)  # [2, 3]
        o1 = torch.matmul(k, q)
        o2 = self.o(q)
        return o1, o2


class ReshapeReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 60)
        self.final_2 = nn.Linear(2, 5)
        self.final_4 = nn.Linear(4, 3)

    def forward(self, x):
        base = self.base(x)
        reshape4 = base.view(1, 4, 15)
        reshape2 = reshape4.view(1, 2, 2, 15)
        o4 = self.final_4(reshape4.permute(0, 2, 1))
        o2 = self.final_2(reshape2.permute(0, 3, 2, 1))
        return o2, o4


@dataclass
class GroupTestDesc:
    model_desc: IModelDesc
    ref_groups: List[PruningNodeGroup]


TEST_DESCS = [
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_builder=SelfAttention,
            input_sample_sizes=([384, 768])
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(64, 0, 1, 0),
                    MinimalDimensionBlock(64, 0, 2, 0),
                    MinimalDimensionBlock(64, 0, 3, 0),
                    MinimalDimensionBlock(64, 0, 17, 1)
                })
        ]
    ),

    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_builder=ReshapeReshape,
            input_sample_sizes=([1, 3])
        ),
        ref_groups=[
            # TODO: ticket 106556: issue with open/closed branches and filtering blocks
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='1_layer_BERT',
            input_info=[dict(sample_size=[1, 10], type='long')] * 3,
            model_builder=partial(
                AutoModelForQuestionAnswering.from_config, BertConfig(num_hidden_layers=1))
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(64, 0, 11, 0),
                    MinimalDimensionBlock(64, 0, 12, 0),
                    MinimalDimensionBlock(64, 0, 15, 0),
                    MinimalDimensionBlock(64, 0, 30, 1),
                }),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(1, 0, 34, 0),
                    MinimalDimensionBlock(1, 0, 36, 1)
                })
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='larger_BERT',
            input_info=[dict(sample_size=[1, 128], type='long')] * 4,
            model_builder=partial(AutoModelForSequenceClassification.from_config,
                                  BertConfig(
                                      hidden_size=4,
                                      intermediate_size=3,
                                      max_position_embeddings=128,
                                      num_attention_heads=2,
                                      num_hidden_layers=1,
                                      vocab_size=10,
                                      num_labels=2,
                                      mhsa_qkv_bias=True,
                                      mhsa_o_bias=True,
                                      ffn_bias=True
                                  ))
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=2, offset=0, producer_id=31, pruning_dimension=1),
                    MinimalDimensionBlock(
                        size=2, offset=0, producer_id=16, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=2, offset=0, producer_id=12, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=2, offset=0, producer_id=13, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=35, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=37, pruning_dimension=1)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=42, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=45, pruning_dimension=1)
                }
            )
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='RoBERTa',
            input_info=[dict(sample_size=[1, 10], type='long')] * 3,
            model_builder=partial(AutoModelForQuestionAnswering.from_config, RobertaConfig(
                num_hidden_layers=1
            ))
        ),
        ref_groups=[
            PruningNodeGroup(dim_blocks={MinimalDimensionBlock(size=64, offset=0, producer_id=38, pruning_dimension=1),
                                         MinimalDimensionBlock(
                                             size=64, offset=0, producer_id=19, pruning_dimension=0),
                                         MinimalDimensionBlock(
                                             size=64, offset=0, producer_id=20, pruning_dimension=0),
                                         MinimalDimensionBlock(size=64, offset=0, producer_id=23,
                                                               pruning_dimension=0)}),
            PruningNodeGroup(dim_blocks={MinimalDimensionBlock(size=1, offset=0, producer_id=44, pruning_dimension=1),
                                         MinimalDimensionBlock(size=1, offset=0, producer_id=42,
                                                               pruning_dimension=0)})
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='Wave2Vec 2.0',
            input_info=dict(sample_size=[1, 400]),
            model_builder=partial(AutoModelForAudioClassification.from_config, Wav2Vec2Config(
                vocab_size=2,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=4,
                conv_dim=(2, 2, 2, 2, 2, 2, 2),
            ))
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=8, offset=0, producer_id=31, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=8, offset=0, producer_id=53, pruning_dimension=1),
                    MinimalDimensionBlock(
                        size=8, offset=0, producer_id=29, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=8, offset=0, producer_id=35, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=57, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=60, pruning_dimension=1)
                }
            )
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='Swin',
            input_info=dict(sample_size=[1, 3, 224, 224]),
            model_builder=partial(AutoModelForImageClassification.from_config,
                                  SwinConfig(
                                      depths=[1],
                                      num_heads=[2],
                                      image_size=224,
                                      patch_size=4,
                                      num_channels=3,
                                      embed_dim=2,
                                  ))
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=15, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=18, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=14, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=33, pruning_dimension=1)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=45, pruning_dimension=1),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=43, pruning_dimension=0)
                }
            )
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='ViT',
            input_info=dict(sample_size=[1, 1, 4, 4]),
            model_builder=partial(AutoModelForImageClassification.from_config,
                                  ViTConfig(
                                      hidden_size=2,
                                      num_hidden_layers=1,
                                      num_attention_heads=2,
                                      intermediate_size=2,
                                      image_size=4,
                                      patch_size=2,
                                      num_channels=1
                                  ))
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=26, pruning_dimension=1),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=9, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=8, pruning_dimension=0),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=12, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=32, pruning_dimension=1),
                    MinimalDimensionBlock(
                        size=1, offset=0, producer_id=30, pruning_dimension=0)
                }
            ),
        ]
    ),
]


@pytest.mark.parametrize(
    "desc", TEST_DESCS, ids=[m.model_desc.model_name for m in TEST_DESCS]
)
def test_graph(desc: GroupTestDesc):
    model_desc = desc.model_desc
    model = model_desc.get_model()
    config = NNCFConfig({"input_info": model_desc.create_input_info()})
    nncf_network = create_nncf_network(model, config)
    pruning_producing_types = [x.op_func_name for x in NNCF_PRUNING_MODULES_DICT]
    actual_groups = get_pruning_groups(nncf_network.get_graph(),
                                       PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES,
                                       pruning_producing_types)

    assert actual_groups == desc.ref_groups
