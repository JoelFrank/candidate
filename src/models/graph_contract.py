"""Canonical U->V loss edges, bidirectional GCN messages (self-loops added by GCN)."""
import torch

def undirected_bipartite(edge_index, edge_weight, num_u):
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('Expected edge_index [2,E]')
    u, v = edge_index
    if edge_index.numel() and not bool(((u >= 0) & (u < num_u) & (v >= num_u)).all()):
        raise ValueError('Pass canonical U->V edges, not already mirrored edges')
    if edge_weight is not None:
        if edge_weight.numel() != u.numel() or not bool(torch.isfinite(edge_weight).all()) or bool((edge_weight < 0).any()):
            raise ValueError('Invalid edge weights')
        edge_weight = torch.cat([edge_weight, edge_weight])
    return torch.cat([edge_index, edge_index.flip(0)], dim=1), edge_weight
