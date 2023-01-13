from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import pytest
import torch
import numpy as np

from torch import nn

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
            PruningNodeGroup({
                MinimalDimensionBlock(64, 0, 1),
                MinimalDimensionBlock(64, 0, 2),
                MinimalDimensionBlock(64, 0, 3)
            })
        ]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name='BERT',
            input_info=[dict(sample_size=[1, 10], type='long')] * 3,
            model_builder=partial(AutoModelForQuestionAnswering.from_config, BertConfig(num_hidden_layers=1))
        ),
        ref_groups=[
            PruningNodeGroup({
                MinimalDimensionBlock(64, 0, 1),
                MinimalDimensionBlock(64, 0, 2),
                MinimalDimensionBlock(64, 0, 3)
            })
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
