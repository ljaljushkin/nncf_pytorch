import sys
import zipfile
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

import graphviz
import openvino as ov

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph import NNCFGraph
from nncf.common.utils.backend import get_backend
from nncf.common.utils.backend import BackendType


@dataclass
class GvNodeDesc:
    node_id: str
    node_type: str
    fillcolor: str
    attributes: List[Tuple[str, str]]

    def build_label(self) -> str:
        new_line = "\l"
        rows = [f"{attr_name}:{attr_value}{new_line}" for attr_name, attr_value in self.attributes]
        rows = "".join(rows)
        begin = "{"
        end = "}"
        label = f"{begin}{self.node_type}|node_id:{self.node_id}{new_line}{rows}{end}"
        return label


@dataclass
class GvEdgeDesc:
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    shape: str


def shape2str(shape: List[int]) -> str:
    s = ", ".join(map(str, shape))
    return f"({s})"


def shape2str_v2(shape: List[int]) -> str:
    return "×".join(map(str, shape))


def build_digraph(node_descs: List[GvNodeDesc], edge_descs: List[GvEdgeDesc]) -> graphviz.Digraph:
    dg = graphviz.Digraph(format="svg")

    # Configure graph attributes
    dg.graph_attr["ranksep"] = "0.2"
    # Configure node attributes
    dg.node_attr["style"] = "rounded,filled"
    dg.node_attr["shape"] = "record"
    dg.node_attr["fontsize"] = "10"
    dg.node_attr["fontname"] = "Tahoma"
    dg.node_attr["fillcolor"] = "white"
    # Configure edge attributes
    dg.edge_attr["arrowsize"] = "0.5"
    dg.edge_attr["fontsize"] = "10"
    dg.edge_attr["fontname"] = "Tahoma"

    for v in node_descs:
        dg.node(v.node_id, label=v.build_label(), fillcolor=v.fillcolor, id=v.node_id)

    for e in edge_descs:
        # dg.edge(e.from_node, e.to_node, label=e.shape)
        dg.edge(e.from_node, e.to_node)

    return dg


def get_graph_desc(
        graph: NNCFGraph,
        fillcolor_map: Dict[str, str],
        default_fillcolor: str = "white"
) -> Tuple[List[GvNodeDesc], List[GvEdgeDesc], Dict[str, Dict[str, str]]]:
    """
    :param graph:
    :param fillcolors:
    :param default_fillcolor:
    :return:
    """
    node_descs: List[GvNodeDesc] = []
    edge_descs: List[GvEdgeDesc] = []
    data: Dict[str, Dict[str, str]] = {}

    for v in graph.get_all_nodes():
        node_id = str(v.node_id)

        node_descs.append(GvNodeDesc(
            node_id=node_id,
            node_type=v.node_type,
            fillcolor=fillcolor_map.get(v.node_type, default_fillcolor),
            attributes=[]
        ))

        node_data = data.setdefault(node_id, {})
        node_data["type"] = v.node_type
        node_data["name"] = v.node_name

    for e in graph.get_all_edges():
        edge_descs.append(GvEdgeDesc(
            from_node=str(e.from_node.node_id),
            from_port=str(e.output_port_id),
            to_node=str(e.to_node.node_id),
            to_port=str(e.input_port_id),
            shape=shape2str_v2(e.tensor_shape),
        ))

    return node_descs, edge_descs, data


def get_fillcolor_map(backend: BackendType):
    if backend == BackendType.OPENVINO:
        fillcolor_map = {
            "FakeQuantize": "#B3C8EA",
            "Add": "#b0bec5",
            "Multiply": "#b0bec5",
            "Matmul": "#81D4FA",
            "Reshape": "#FFB74D",
            "Parameter": "#EA80FC",
            "CollapsedEdges": "#DCEDC8"
        }
        return fillcolor_map
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def apply_constant_hiding(
    node_descs: List[GvNodeDesc],
    edge_descs: List[GvEdgeDesc],
    graph: NNCFGraph,
    backend: BackendType,
) -> Tuple[List[GvNodeDesc], List[GvEdgeDesc]]:
    """
    :param node_descs:
    :param edge_descs:
    :param graph:
    :param backend:
    :return:
    """
    if backend == BackendType.OPENVINO:
        from nncf.openvino.graph.metatypes import openvino_metatypes
        constant_metatype = openvino_metatypes.OVConstantMetatype
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    constant_nodes_to_hide: List[str] = []  # node_id
    node_id_to_constant_inputs: Dict[str, List[Tuple[str, str]]] = {}  # {node_id: [(input_port_id, constant_shape)]}

    constant_nodes = graph.get_nodes_by_metatypes([constant_metatype])
    for constant_node in constant_nodes:
        in_edges = graph.get_input_edges(constant_node)
        assert len(in_edges) == 0

        out_edges = graph.get_output_edges(constant_node)
        if len(out_edges) > 1:
            continue
        assert len(out_edges) != 0
        edge = out_edges[0]
        assert edge.from_node.node_id == constant_node.node_id

        constant_nodes_to_hide.append(str(constant_node.node_id))

        consumer_id = str(edge.to_node.node_id)
        inputs = node_id_to_constant_inputs.setdefault(consumer_id, [])
        inputs.append((str(edge.input_port_id), shape2str(edge.tensor_shape)))

    for node_id in node_id_to_constant_inputs:
        node_id_to_constant_inputs[node_id].sort(key=lambda x: int(x[0]))

    # Remove constant nodes
    new_node_descs = [
        node_desc for node_desc in node_descs if node_desc.node_id not in constant_nodes_to_hide
    ]

    # TODO: Improve it
    # Add attributes
    for node_id, constant_inputs in node_id_to_constant_inputs.items():
        for node_desc in new_node_descs:
            if node_desc.node_id != node_id:
                continue
            for input_port_id, constant_shape in constant_inputs:
                node_desc.attributes.append((input_port_id, constant_shape))

    # Remove edges from constant nodes
    new_edge_descs = [
        edge for edge in edge_descs if edge.from_node not in constant_nodes_to_hide
    ]

    return new_node_descs, new_edge_descs


def apply_edge_collapsing(
    node_descs: List[GvNodeDesc],
    edge_descs: List[GvEdgeDesc],
    graph: NNCFGraph,
    allowed_outdegree: int = 10,
    num_visible: int = 2,
):
    node_id_and_outdegree = [
        (v.node_id, len(graph.get_next_nodes(v))) for v in graph.get_all_nodes()
    ]
    node_id_and_outdegree.sort(key=lambda x: x[1], reverse=True)

    # We should collapse edges for these nodes
    node_ids = [node_id for node_id, outdegree in node_id_and_outdegree if outdegree > allowed_outdegree]

    topological_sorted_node_ids = [v.node_id for v in graph.topological_sort()]
    assert len(topological_sorted_node_ids) == len(set(topological_sorted_node_ids))
    key_fn = lambda x: topological_sorted_node_ids.index(int(x.to_node))

    node_id_to_edges: Dict[str, List[GvEdgeDesc]] = {}

    for node_id in node_ids:
        node = graph.get_node_by_id(node_id)
        out_edges = graph.get_output_edges(node)
        for e in out_edges:
            assert e.from_node.node_id == node_id

            edges = node_id_to_edges.setdefault(str(node_id), [])
            edges.append(GvEdgeDesc(
                from_node=str(e.from_node.node_id),
                from_port=str(e.output_port_id),
                to_node=str(e.to_node.node_id),
                to_port=str(e.input_port_id),
                shape=shape2str_v2(e.tensor_shape),
            ))

        node_id_to_edges[str(node_id)].sort(key=key_fn)


    edges_to_add = []
    edges_to_remove = []

    # Add GvNodeDesc for CollapsedEdges nodes
    for idx, (node_id, edges) in enumerate(node_id_to_edges.items()):
        collapsed_edge_node_attrs = []

        for i, e in enumerate(edges):
            if i < num_visible:
                continue

            collapsed_edge_node_attrs.append(("node_id", e.to_node))
            edges_to_remove.append(e)

        collapsed_edge_node = GvNodeDesc(
            node_id=f"ce_{idx}",
            node_type="CollapsedEdges",
            fillcolor="#DCEDC8",
            attributes=collapsed_edge_node_attrs,
        )
        node_descs.append(collapsed_edge_node)

        edges_to_add.append(GvEdgeDesc(
            from_node=node_id,
            from_port="",
            to_node=f"ce_{idx}",
            to_port="",
            shape=""
        ))

    new_edge_descs = []
    for x in edge_descs:
        should_add = True
        for e in edges_to_remove:
            if x == e:
                should_add = False
        if should_add:
            new_edge_descs.append(x)

    new_edge_descs.extend(edges_to_add)

    return node_descs, new_edge_descs


def add_ov_attributes(model: ov.Model, graph: NNCFGraph, data):
    friendly_name_to_op = {
        op.get_friendly_name(): op for op in model.get_ops()
    }

    for node in graph.get_all_nodes():
        data[str(node.node_id)]["op_attributes"] = friendly_name_to_op[node.node_name].get_attributes()


def save_for_ngviewer(model, filename: str) -> None:
    backend = get_backend(model)
    graph = NNCFGraphFactory.create(model)

    fillcolor_map = get_fillcolor_map(backend)
    node_descs, edge_descs, data = get_graph_desc(graph, fillcolor_map)

    if backend == BackendType.OPENVINO:
        # Add attributes from ov.Model
        add_ov_attributes(model, graph, data)

    # ========== Apply passes ==========
    node_descs, edge_descs = apply_constant_hiding(node_descs, edge_descs, graph, backend)
    node_descs, edge_descs = apply_edge_collapsing(node_descs, edge_descs, graph)
    # ========== Apply passes ==========

    dg = build_digraph(node_descs, edge_descs)
    dg.render("graph", cleanup=True)

    # ========== Save to .NG file ==========
    with zipfile.ZipFile(filename, mode="w") as ng_archive:
        ng_archive.write("graph.svg", arcname="graph.svg")
        with ng_archive.open("data.json", mode="w") as f:
            json_str = json.dumps(data, indent=4)
            f.write(json_str.encode())

    with zipfile.ZipFile(filename, mode="r") as ng_archive:
        ng_archive.printdir()
    # ========== Save to .NG file ==========


def run_app(model_path: str, output_path: str):
    print("========== Save for NG viewer ==========")
    print("Input Model: ", model_path)
    print("Output: ", output_path)

    model = ov.Core().read_model(model_path)
    save_for_ngviewer(model, output_path)


if __name__ == "__main__":
    run_app(sys.argv[1], sys.argv[2])