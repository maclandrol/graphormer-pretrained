from typing import Optional
from typing import List
import torch
from graphormer_pretrained.data.collator import collator
from graphormer_pretrained.data.smiles_dataset import GraphormerInferenceDataset
from graphormer_pretrained.models.graphormer import GraphormerModel
from graphormer_pretrained.models.graphormer import GraphormerEncoder
from graphormer_pretrained.models.graphormer import graphormer_base_architecture
from graphormer_pretrained.utils.common import safe_hasattr
from graphormer_pretrained.tasks.graph_prediction import GraphPredictionConfig


class GraphormerEmbeddingsExtractor:
    """
    Extract embeddings from a graphormer pretrained model
    """

    def __init__(
        self,
        pretrained_name: str = "pcqm4mv1_graphormer_base",
        max_nodes: Optional[int] = None,
    ):
        self.pretrained_name = pretrained_name
        self.config = GraphPredictionConfig()
        if max_nodes is not None:
            self.config.max_nodes = max_nodes
        self.config.pretrained_model_name = self.pretrained_name
        self.config.remove_head = True
        self.encoder = None
        self.model = None
        self._load()

    def _load(self):
        """Load and initialize a pretrained graphormer model for molecular embedding"""
        graphormer_base_architecture(self.config)
        if not safe_hasattr(self.config, "max_nodes"):
            self.config.max_nodes = self.config.tokens_per_sample
        self.encoder = GraphormerEncoder(self.config)
        self.model = GraphormerModel(self.config, self.encoder)
        self.model.eval()

    def _convert(self, smiles: List[str]):
        """Convert a list of input structure into"""
        dataset = GraphormerInferenceDataset(
            smiles,
            multi_hop_max_dist=self.config.multi_hop_max_dist,
            spatial_pos_max=self.config.spatial_pos_max,
        )
        batch = collator(
            [dataset[i] for i in range(len(dataset))],
            max_node=self.config.max_nodes,
            multi_hop_max_dist=self.config.multi_hop_max_dist,
            spatial_pos_max=self.config.spatial_pos_max,
            ignore_large_graph=False,
        )
        del dataset
        return batch

    @torch.no_grad()
    def __call__(self, smiles: List[str]):
        """Predict molecular embeddings from a list of SMILES strings"""

        batch_graphs = self._convert(smiles)
        return self.model(batch_graphs)
