from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import pytest
import torch
# import nncf.torch  # TODO: why "import nncf" is not enough for calling patch_torch_operators??
from transformers import AutoModelForImageClassification
from transformers import DistilBertConfig
from transformers import RobertaConfig
from transformers import SwinConfig
from transformers import ViTConfig

from nncf.torch import patch_torch_operators
from tests.torch.test_models.swin import SwinTransformerBlock

patch_torch_operators()  # more reliable, since it's not removed as unused

from torch import nn
# TODO: why "import nncf" is not enough for calling patch_torch_operators??
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import CLIPVisionConfig
from transformers import CLIPVisionModel
from transformers import GPT2Config
from transformers import MobileBertConfig
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


# TODO: why this import breaks BERT tests???
#             assert len(output_tensors_shapes) == 1 or len(set(output_tensors_shapes)) <= 1, node.node_name
#  >           output_tensors_shape = output_tensors_shapes[0]
#  E           IndexError: list index out of range
#  need to dump graph, seems like some operation is not wrapped because of order of imports?
#  ANSWER: gelu is not patched, graph is disjointed and linear has no output shapes, which is not expected.


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
            model_name='1_layer_BERT',
            input_info=[dict(sample_size=[1, 10], type='long')] * 3,
            model_builder=partial(AutoModelForQuestionAnswering.from_config, BertConfig(num_hidden_layers=1))
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
            # input_info = [
            #     ModelInputInfo(shape=[1, 128], type_str='long', keyword='input_ids'),
            #     ModelInputInfo(shape=[1, 128], type_str='long', keyword='attention_mask'),
            #     ModelInputInfo(shape=[1, 128], type_str='long', keyword='token_type_ids'),
            #     ModelInputInfo(shape=[1, 128], type_str='long', keyword='position_ids'),
            # ],
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
                    MinimalDimensionBlock(size=2, offset=0, producer_id=31, pruning_dimension=1),
                    MinimalDimensionBlock(size=2, offset=0, producer_id=16, pruning_dimension=0),
                    MinimalDimensionBlock(size=2, offset=0, producer_id=12, pruning_dimension=0),
                    MinimalDimensionBlock(size=2, offset=0, producer_id=13, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=35, pruning_dimension=0),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=37, pruning_dimension=1)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=42, pruning_dimension=0),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=45, pruning_dimension=1)
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
                                         MinimalDimensionBlock(size=64, offset=0, producer_id=19, pruning_dimension=0),
                                         MinimalDimensionBlock(size=64, offset=0, producer_id=20, pruning_dimension=0),
                                         MinimalDimensionBlock(size=64, offset=0, producer_id=23,
                                                               pruning_dimension=0)}),
            PruningNodeGroup(dim_blocks={MinimalDimensionBlock(size=1, offset=0, producer_id=44, pruning_dimension=1),
                                         MinimalDimensionBlock(size=1, offset=0, producer_id=42,
                                                               pruning_dimension=0)})
        ]
    ),
    # TODO: KeyError: 'output_mask'. Probably because of attention masks on before Softmax

    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='DistilBERT',
            input_info=[dict(sample_size=[1, 4], type='long')] * 2,
            model_builder=partial(
                AutoModelForQuestionAnswering.from_config,
                DistilBertConfig(
                    vocab_size=4,
                    max_position_embeddings=4,
                    n_layers=1,
                    n_heads=1,
                    dim=4,
                    hidden_dim=4 * 4,
                )
            )
        ),
        ref_groups=[
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='MobileBERT',
            input_info=[dict(sample_size=[1, 128], type='long')] * 4,
            model_builder=partial(AutoModelForSequenceClassification.from_config,
                                  MobileBertConfig(
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
            # TODO: Why was this group filtered out?
            # PruningNodeGroup(
            #     dim_blocks={
            #         MinimalDimensionBlock(size=1, offset=0, producer_id=22, pruning_dimension=0),
            #         MinimalDimensionBlock(size=1, offset=0, producer_id=24, pruning_dimension=1),
            #         MinimalDimensionBlock(size=1, offset=0, producer_id=25, pruning_dimension=1),
            #     }
            # ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=64, offset=0, producer_id=26, pruning_dimension=0),
                    MinimalDimensionBlock(size=64, offset=0, producer_id=27, pruning_dimension=0),
                    MinimalDimensionBlock(size=64, offset=0, producer_id=25, pruning_dimension=0),
                    MinimalDimensionBlock(size=64, offset=0, producer_id=44, pruning_dimension=1)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=50, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=48, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=56, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=54, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=62, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=60, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=68, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=66, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=81, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=78, pruning_dimension=0)
                }
            )
        ]
    ),
    # TODO: need to handle model_output in the middle of each Transformer layer
    # TODO: note, that Split before Reshape, but each branch has the same Reshape - should be fine
    # TODO: support Addmm properly, attributes?
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='GPT2Text',
            input_info=[dict(sample_size=[1, 128], type='long')] * 1,
            model_builder=partial(AutoModelForSequenceClassification.from_config,
                                  GPT2Config(n_embd=4, n_layer=2, n_head=2, vocab_size=2))
        ),
        ref_groups=[]
    ),
    # TODO: need to handle concat with constant
    # TODO: pay attention to transpose that does not change shape: 1x2x2x1 -> 1x2x2x1
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='CLIP',
            input_info=[dict(sample_size=[1, 3, 3, 3], type='float')] * 1,
            model_builder=partial(CLIPVisionModel,
                                  CLIPVisionConfig(
                                      hidden_size=2,
                                      intermediate_size=2,
                                      num_hidden_layers=1,
                                      num_attention_heads=2,
                                      num_channels=3,
                                      image_size=3,
                                      patch_size=3,
                                  ))
        ),
        ref_groups=[]
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
                    MinimalDimensionBlock(size=8, offset=0, producer_id=31, pruning_dimension=0),
                    MinimalDimensionBlock(size=8, offset=0, producer_id=53, pruning_dimension=1),
                    MinimalDimensionBlock(size=8, offset=0, producer_id=29, pruning_dimension=0),
                    MinimalDimensionBlock(size=8, offset=0, producer_id=35, pruning_dimension=0)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=57, pruning_dimension=0),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=60, pruning_dimension=1)
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
                    MinimalDimensionBlock(size=1, offset=0, producer_id=15, pruning_dimension=0),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=18, pruning_dimension=0),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=14, pruning_dimension=0),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=33, pruning_dimension=1)
                }
            ),
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=45, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=43, pruning_dimension=0)
                }
            )
        ]
    ),
    # TODO: not empty mask on concat with Constant is not supported
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

        ]
    ),
    # TODO: reshape for more than 2 dims is not supported
    # TODO: need a symbolic propagation
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='Swin_MS',
            input_info=dict(sample_size=[1, 4 * 4, 8]),
            model_builder=partial(SwinTransformerBlock, dim=8, input_resolution=[4, 4], num_heads=2)
        ),
        ref_groups=[
            PruningNodeGroup(
                dim_blocks={
                    MinimalDimensionBlock(size=1, offset=0, producer_id=36, pruning_dimension=1),
                    MinimalDimensionBlock(size=1, offset=0, producer_id=33, pruning_dimension=0)
                }
            )
        ]
    )
]


@pytest.mark.parametrize(
    "desc", TEST_DESCS, ids=[m.model_desc.model_name for m in TEST_DESCS]
)
def test_graph(desc: GroupTestDesc):
    model_desc = desc.model_desc
    model = model_desc.get_model()
    config = NNCFConfig({"input_info": model_desc.create_input_info()})
    nncf_network = create_nncf_network(model, config)
    nncf_network.get_graph().visualize_graph('nncf_network.dot')
    pruning_producing_types = [x.op_func_name for x in NNCF_PRUNING_MODULES_DICT]
    actual_groups = get_pruning_groups(nncf_network.get_graph(),
                                       PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES,
                                       pruning_producing_types)

    assert actual_groups == desc.ref_groups
