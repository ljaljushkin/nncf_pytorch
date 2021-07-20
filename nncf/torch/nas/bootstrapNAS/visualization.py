import networkx as nx

from nncf.torch.layers import NNCFConv2d
from nncf.torch.nas.bootstrapNAS.algo import BootstrapNASController
from nncf.torch.nncf_network import NNCFNetwork


class WidthGraph:
    def __init__(self, model: NNCFNetwork, compression_ctrl: BootstrapNASController = None):
        nncf_graph = model.get_graph()
        width_list = []
        self._nx_graph = nncf_graph.get_graph_for_structure_analysis()
        node_name_vs_elastic_op_and_groip_id_map = dict()
        if compression_ctrl is not None:
            for cluster in compression_ctrl.pruned_module_groups_info.get_all_clusters():
                for element in cluster.elements:
                    node_name_vs_elastic_op_and_groip_id_map[element.node_name] = (element.elastic_op, cluster.id)
        for node_key in nncf_graph.get_all_node_keys():
            node = nncf_graph.get_node_by_key(node_key)
            color = ''
            operator_name = node.node_type
            node_name = node.node_name
            module = model.get_containing_module(node.node_name)
            if isinstance(module, NNCFConv2d):
                elastic_op, group_id = node_name_vs_elastic_op_and_groip_id_map.get(node_name, (None, None))
                in_width = module.in_channels if elastic_op is None else '???'
                out_width = module.out_channels if elastic_op is None else elastic_op.active_num_out_channels
                width_list.append(out_width)
                color = 'lightblue'
                if module.groups == module.in_channels and module.in_channels > 1:
                    operator_name = 'DW_Conv2d'
                    color = 'purple'
                operator_name += f'_{in_width}x{out_width}'
                if group_id is not None:
                    operator_name += f'_G{group_id}'
            operator_name += '_#{}'.format(node.node_id)
            target_node_to_draw = self._nx_graph.nodes[node_key]
            target_node_to_draw['label'] = operator_name
            target_node_to_draw['style'] = 'filled'
            if color:
                target_node_to_draw['color'] = color

    def get(self) -> nx.DiGraph:
        return self._nx_graph
