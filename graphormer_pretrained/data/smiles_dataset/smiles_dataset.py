# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
import torch
import numpy as np

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from graphormer_pretrained.data.wrapper import preprocess_item
from graphormer_pretrained.data import algos
from graphormer_pretrained.data.pyg_datasets import GraphormerPYGDataset

from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data as PYGGraph
from torch_geometric.transforms.to_dense import ToDense


class GraphormerSMILESDataset(GraphormerPYGDataset):
    def __init__(
        self,
        dataset: str,
        num_class: int,
        max_node: int,
        multi_hop_max_dist: int,
        spatial_pos_max: int,
    ):
        self.dataset = np.genfromtxt(dataset, delimiter=",", dtype=str)
        num_data = len(self.dataset)
        self.num_class = num_class
        self.__get_graph_metainfo(max_node, multi_hop_max_dist, spatial_pos_max)
        train_valid_idx, test_idx = train_test_split(num_data // 10)
        train_idx, valid_idx = train_test_split(train_valid_idx, num_data // 5)
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.__indices__ = None
        self.train_data = self.index_select(train_idx)
        self.valid_data = self.index_select(valid_idx)
        self.test_data = self.index_select(test_idx)

    def __get_graph_metainfo(self, max_node: int, multi_hop_max_dist: int, spatial_pos_max: int):
        self.max_node = min(
            max_node,
            torch.max(self.dataset[i][0].num_nodes() for i in range(len(self.dataset))),
        )
        max_dist = 0
        for i in range(len(self.dataset)):
            pyg_graph = smiles2graph(self.dataset[i])
            dense_adj = pyg_graph.adj().to_dense().type(torch.int)
            shortest_path_result, _ = algos.floyd_warshall(dense_adj.numpy())
            max_dist = max(max_dist, np.amax(shortest_path_result))
        self.multi_hop_max_dist = min(multi_hop_max_dist, max_dist)
        self.spatial_pos_max = min(spatial_pos_max, max_dist)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = smiles2graph(self.dataset[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")


class GraphormerInferenceDataset(GraphormerSMILESDataset):
    """Inference dataset for graphormer"""

    def __init__(self, smiles: List[str], multi_hop_max_dist: int, spatial_pos_max: int):
        self.dataset = smiles
        self.num_class = 1
        self.preprocessed_dataset = self._build_graph_info(multi_hop_max_dist, spatial_pos_max)

    def _build_graph_info(self, multi_hop_max_dist: int, spatial_pos_max: int):
        max_dist = 0
        preprocessed_dataset = []
        for smiles in tqdm(self.dataset, desc="Preprocessing Data", leave=False):
            pyg_graph = PYGGraph()
            graph = smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            pyg_graph.__num_nodes__ = int(graph["num_nodes"])
            pyg_graph.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            pyg_graph.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            pyg_graph.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            pyg_graph.y = torch.zeros([1])
            item = preprocess_item(pyg_graph)
            preprocessed_dataset.append(item)
            max_dist = max(max_dist, item.edge_input.shape[0])
        self.multi_hop_max_dist = min(multi_hop_max_dist, max_dist)
        self.spatial_pos_max = min(spatial_pos_max, max_dist)
        return preprocessed_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.preprocessed_dataset[idx]
            item.idx = idx
            return item
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")
