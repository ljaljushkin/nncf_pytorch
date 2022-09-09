import functools
from typing import Dict, Callable, Any, Union, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos, create_dummy_forward_fn
from nncf.torch import create_compressed_model
from nncf import NNCFConfig
from nncf.torch.initialization import register_default_init_args
from abc import ABC, abstractmethod
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks, get_building_blocks_info

from transformers import AutoModelForQuestionAnswering

class BaseDatasetMock(Dataset, ABC):
    def __init__(self, input_size: Tuple, num_samples: int = 10):
        super().__init__()
        self._input_size = input_size
        self._len = num_samples

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __len__(self) -> int:
        return self._len

class RandomDatasetMock(BaseDatasetMock):
    def __getitem__(self, index):
        return torch.rand(self._input_size), torch.zeros(1)

def create_any_mock_dataloader(dataset_cls: type, config: NNCFConfig, num_samples: int = 1,
                               batch_size: int = 1) -> DataLoader:
    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = DataLoader(dataset_cls(input_sample_size[1:], num_samples),
                             batch_size=batch_size,
                             num_workers=0,  # Workaround
                             shuffle=False, drop_last=True)
    return data_loader

create_random_mock_dataloader = functools.partial(create_any_mock_dataloader, dataset_cls=RandomDatasetMock)

bs=1
nbatch = 3200 # dummy
seqlen = 384

model_path = 'bert-base-uncased'
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

nncf_cfgdict = dict(
    input_info = [dict(sample_size=[bs, seqlen], type='long')]*3,
)
nncf_cfg = NNCFConfig.from_dict(nncf_cfgdict)

mock_dataloader = create_random_mock_dataloader(config=nncf_cfg, num_samples=nbatch*bs, batch_size=bs)
nncf_cfg = register_default_init_args(nncf_cfg, mock_dataloader)

nncf_ctrl, nncf_model = create_compressed_model(model, nncf_cfg)

blocks, _, group_dependent = get_building_blocks(nncf_model, allow_nested_blocks=False)
blocks_info = get_building_blocks_info(blocks, nncf_model)

DETAIL=False
g=nncf_model.get_graph()
for i, bbi in enumerate(blocks_info):
    op_addresses_ids = list(map(lambda x: g.get_node_by_name(x.__str__()).node_id, bbi.op_addresses))
    print("\n- {} | {} ---".format(i, bbi.block_type))
    print("start  : {}\nend    : {}\nNodeIDs:{}\n|\n".format(
        bbi.building_block.start_node_name,
        bbi.building_block.end_node_name,
        op_addresses_ids))

    if DETAIL is True:
        for oi, opaddr in enumerate(bbi.op_addresses):
            print("NodeID {:3}: {}".format(g.get_node_by_name(opaddr.__str__()).node_id, opaddr))