from transformers import AutoModelForAudioClassification

from nncf import NNCFConfig
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks, get_building_blocks_info
from nncf.torch import create_compressed_model

bs = 1
n_sample = 400

model_path = 'anton-l/wav2vec2-base-ft-keyword-spotting'
model = AutoModelForAudioClassification.from_pretrained(model_path)

nncf_cfgdict = dict(
    input_info=dict(sample_size=[bs, n_sample]),
)
nncf_cfg = NNCFConfig.from_dict(nncf_cfgdict)

nncf_ctrl, nncf_model = create_compressed_model(model, nncf_cfg)

nncf_model.eval()
ext_blocks, group_dependent = get_building_blocks(nncf_model)
blocks_info = get_building_blocks_info(ext_blocks, nncf_model)

DETAIL = False
g = nncf_model.get_graph()
for i, bbi in enumerate(blocks_info):
    op_addresses_ids = list(map(lambda x: g.get_node_by_name(x.__str__()).node_id, bbi.op_addresses))
    print("\n- {} | {} ---".format(i, bbi.block_type))
    print("start  : {}\nend    : {}\nNodeIDs:{}\n|\n".format(
        bbi.building_block.start_node_name,
        bbi.building_block.end_node_name,
        sorted(op_addresses_ids)))

    if DETAIL is True:
        for oi, opaddr in enumerate(bbi.op_addresses):
            print("NodeID {:3}: {}".format(g.get_node_by_name(opaddr.__str__()).node_id, opaddr))
