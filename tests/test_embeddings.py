import datamol as dm
import torch
from graphormer_pretrained.embeddings import GraphormerEmbeddingsExtractor


def test_embeddings():
    model = GraphormerEmbeddingsExtractor()
    smiles = dm.data.freesolv()["smiles"].sample(n=100).values
    embeddings = model(smiles)

    model2 = GraphormerEmbeddingsExtractor(max_nodes=25)
    embeddings2 = model2(smiles)
    assert embeddings.shape[0] == embeddings2.shape[0], "Unexpected embedding number"
    assert embeddings.shape[-1] == embeddings2.shape[-1], "Unexpected embedding shape"
    assert len(embeddings.shape) == 3, "Expected 3 dim tensor"
    assert torch.is_tensor(embeddings), f"Expected tensor, got type {type(embeddings)}"
