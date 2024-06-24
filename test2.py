import networkx as nx
from collections import defaultdict
from pyvis.network import Network
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
from src.rolx import RecursiveFeatureExtractor, RoleExtractor


dataset = Planetoid(root='datasets/Planetoid', name="Cora", transform=NormalizeFeatures())
print()
print(f'Dataset:{dataset}')
print(f"Number of graphs:{len(dataset)}")
print(f"Number of features:{dataset.num_features}")
print(f"Number of classes:{dataset.num_classes}")
data = dataset[0]

print()
print(data)
print("=" * 10)
# Information about the first graph
print(f"Number of nodes:{data.num_nodes}")
print(f"Number of edges:{data.num_edges}")
print(f"Number of training nodes:{data.train_mask.sum()}")
print(f"Number of test nodes:{data.test_mask.sum()}")
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


def generate_role_graph(roles_dic:dict,with_prop:None):
    graph_roles = nx.Graph()
    for key in roles_dic.keys():
        graph_roles.add_node(key)
    values_to_keys = {}
    for key, value in roles_dic.items():
        if value not in values_to_keys:
            values_to_keys[value] = []
        values_to_keys[value].append(key)

    for nodes in values_to_keys.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                graph_roles.add_edge(nodes[i], nodes[j])
    return graph_roles

nx_data = to_networkx(data,to_undirected=True)
print(nx_data)

feature_extractor = RecursiveFeatureExtractor(nx_data)
features = feature_extractor.extract_features()
print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')
print(features)