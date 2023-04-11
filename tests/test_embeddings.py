import datamol as dm
import torch
from graphormer_pretrained.embeddings import GraphormerEmbeddingsExtractor


def test_embeddings():
    model = GraphormerEmbeddingsExtractor()
    smiles = dm.data.freesolv()["smiles"].sample(n=100).values
    embeddings, graph_rep, padding = model(smiles)

    # model2 = GraphormerEmbeddingsExtractor(max_nodes=25)
    # embeddings2, graph_rep2, padding2 = model2(smiles)

    model3 = GraphormerEmbeddingsExtractor(max_nodes=25, concat_layers=[-1, -2])
    embeddings3, graph_rep3, padding3 = model3(smiles)

    assert embeddings.shape[0] == embeddings3.shape[0], "Unexpected embedding number"
    assert embeddings.shape[-1] * 2 == embeddings3.shape[-1], "Unexpected embedding shape"
    assert len(embeddings.shape) == 3, "Expected 3 dim tensor"
    assert torch.is_tensor(embeddings), f"Expected tensor, got type {type(embeddings)}"
    assert graph_rep.shape == graph_rep3.shape
    assert torch.allclose(padding.sum(dim=-1), padding3.sum(dim=-1))
    assert torch.allclose(graph_rep, graph_rep3), "Graph representation do not match"
