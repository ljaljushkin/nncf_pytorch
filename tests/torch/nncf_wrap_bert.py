from transformers import AutoModelForQuestionAnswering

from nncf import NNCFConfig
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks, get_building_blocks_info
from nncf.torch import create_compressed_model

bs = 1
seqlen = 10

model_path = 'bert-base-uncased'
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

nncf_cfgdict = dict(
    input_info=[dict(sample_size=[bs, seqlen], type='long')] * 3,
)
nncf_cfg = NNCFConfig.from_dict(nncf_cfgdict)

nncf_ctrl, nncf_model = create_compressed_model(model, nncf_cfg)

blocks, _, group_dependent = get_building_blocks(nncf_model, allow_nested_blocks=False)
blocks_info = get_building_blocks_info(blocks, nncf_model)

DETAIL = False
g = nncf_model.get_graph()
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
