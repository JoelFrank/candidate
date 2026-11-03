"""
Data splitting utilities for inductive evaluation.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from .dataset import BipartiteDataset
import random


def create_inductive_splits(
    dataset: BipartiteDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    observed_ratio: float = 0.7,
    seed: int = 42
) -> Dict[str, BipartiteDataset]:
    """
    Create inductive splits for bipartite graphs.
    
    This creates a realistic inductive setting where:
    1. Some nodes are only seen during training (observed)
    2. Some nodes are only seen during testing (unobserved)
    3. Edges are split across train/val/test
    
    Args:
        dataset: Original bipartite dataset
        train_ratio: Ratio of edges for training
        val_ratio: Ratio of edges for validation
        test_ratio: Ratio of edges for testing
        observed_ratio: Ratio of nodes seen during training
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    assert 0 < observed_ratio < 1
    
    # Split nodes into observed/unobserved
    num_observed_u = int(dataset.num_nodes_u * observed_ratio)
    num_observed_v = int(dataset.num_nodes_v * observed_ratio)
    
    # Randomly select observed nodes
    observed_u = torch.randperm(dataset.num_nodes_u)[:num_observed_u]
    observed_v = torch.randperm(dataset.num_nodes_v)[:num_observed_v]
    
    # Adjust observed_v indices to global indexing
    observed_v_global = observed_v + dataset.num_nodes_u
    
    # Create node mappings
    unobserved_u = torch.tensor([i for i in range(dataset.num_nodes_u) if i not in observed_u])
    unobserved_v = torch.tensor([i for i in range(dataset.num_nodes_v) if i not in observed_v])
    unobserved_v_global = unobserved_v + dataset.num_nodes_u
    
    observed_nodes = torch.cat([observed_u, observed_v_global])
    unobserved_nodes = torch.cat([unobserved_u, unobserved_v_global])
    
    # Categorize edges based on node visibility
    edge_index = dataset.edge_index
    sources = edge_index[0]
    targets = edge_index[1]
    
    # Check which nodes are observed
    source_observed = torch.isin(sources, observed_nodes)
    target_observed = torch.isin(targets, observed_nodes)
    
    # Edge categories:
    # 1. Both nodes observed (can be in train/val)
    # 2. One node observed, one unobserved (test only)
    # 3. Both nodes unobserved (test only)
    both_observed = source_observed & target_observed
    mixed_edges = source_observed ^ target_observed  # XOR
    both_unobserved = ~source_observed & ~target_observed
    
    # Get edge indices for each category
    train_val_candidates = torch.where(both_observed)[0]
    test_only_mixed = torch.where(mixed_edges)[0]
    test_only_unobserved = torch.where(both_unobserved)[0]
    
    # Split train_val_candidates into train and val
    num_train_val = len(train_val_candidates)
    num_train = int(num_train_val * train_ratio / (train_ratio + val_ratio))
    
    perm = torch.randperm(num_train_val)
    train_indices = train_val_candidates[perm[:num_train]]
    val_indices = train_val_candidates[perm[num_train:]]
    
    # All test edges (mixed + unobserved)
    test_indices = torch.cat([test_only_mixed, test_only_unobserved])
    
    # Create negative edges for evaluation
    train_neg_edges = _sample_negative_edges(
        dataset, train_indices, len(train_indices), observed_nodes
    )
    val_neg_edges = _sample_negative_edges(
        dataset, val_indices, len(val_indices), observed_nodes
    )
    test_neg_edges = _sample_negative_edges(
        dataset, test_indices, len(test_indices), None  # Can use all nodes for test
    )
    
    # Create split datasets
    splits = {}
    
    # Training split (only observed nodes and their edges)
    train_edge_index = edge_index[:, train_indices]
    train_neg_edge_index = train_neg_edges
    
    splits['train'] = {
        'pos_edge_index': train_edge_index,
        'neg_edge_index': train_neg_edge_index,
        'x_u': dataset.x_u[observed_u],
        'x_v': dataset.x_v[observed_v],
        'num_nodes_u': len(observed_u),
        'num_nodes_v': len(observed_v),
        'node_map_u': observed_u,
        'node_map_v': observed_v,
        'observed_nodes': observed_nodes
    }
    
    # Validation split
    val_edge_index = edge_index[:, val_indices]
    val_neg_edge_index = val_neg_edges
    
    splits['val'] = {
        'pos_edge_index': val_edge_index,
        'neg_edge_index': val_neg_edge_index,
        'x_u': dataset.x_u[observed_u],
        'x_v': dataset.x_v[observed_v],
        'num_nodes_u': len(observed_u),
        'num_nodes_v': len(observed_v),
        'node_map_u': observed_u,
        'node_map_v': observed_v,
        'observed_nodes': observed_nodes
    }
    
    # Test split (all nodes, test edges)
    test_edge_index = edge_index[:, test_indices]
    test_neg_edge_index = test_neg_edges
    
    splits['test'] = {
        'pos_edge_index': test_edge_index,
        'neg_edge_index': test_neg_edge_index,
        'x_u': dataset.x_u,
        'x_v': dataset.x_v,
        'num_nodes_u': dataset.num_nodes_u,
        'num_nodes_v': dataset.num_nodes_v,
        'node_map_u': torch.arange(dataset.num_nodes_u),
        'node_map_v': torch.arange(dataset.num_nodes_v),
        'observed_nodes': observed_nodes,
        'unobserved_nodes': unobserved_nodes
    }
    
    return splits


def _sample_negative_edges(
    dataset: BipartiteDataset,
    pos_edge_indices: torch.Tensor,
    num_negative: int,
    allowed_nodes: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Sample negative edges for evaluation.
    
    Args:
        dataset: Bipartite dataset
        pos_edge_indices: Indices of positive edges
        num_negative: Number of negative edges to sample
        allowed_nodes: Nodes that can be used (None = all nodes)
        
    Returns:
        Negative edge indices [2, num_negative]
    """
    # Get existing edges as set for fast lookup
    pos_edges = dataset.edge_index[:, pos_edge_indices]
    existing_edges = set()
    for i in range(pos_edges.size(1)):
        edge = (pos_edges[0, i].item(), pos_edges[1, i].item())
        existing_edges.add(edge)
        existing_edges.add((edge[1], edge[0]))  # Add reverse edge
    
    # Determine node ranges
    if allowed_nodes is not None:
        allowed_u = allowed_nodes[allowed_nodes < dataset.num_nodes_u]
        allowed_v = allowed_nodes[allowed_nodes >= dataset.num_nodes_u]
    else:
        allowed_u = torch.arange(dataset.num_nodes_u)
        allowed_v = torch.arange(dataset.num_nodes_u, dataset.num_nodes_u + dataset.num_nodes_v)
    
    # Sample negative edges
    negative_edges = []
    max_attempts = num_negative * 10  # Prevent infinite loop
    attempts = 0
    
    while len(negative_edges) < num_negative and attempts < max_attempts:
        # Sample random u and v nodes
        u_idx = allowed_u[torch.randint(0, len(allowed_u), (1,))].item()
        v_idx = allowed_v[torch.randint(0, len(allowed_v), (1,))].item()
        
        edge = (u_idx, v_idx)
        if edge not in existing_edges:
            negative_edges.append(edge)
        
        attempts += 1
    
    if len(negative_edges) < num_negative:
        print(f"Warning: Could only sample {len(negative_edges)} negative edges out of {num_negative}")
    
    # Convert to tensor
    if len(negative_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    neg_edge_tensor = torch.tensor(negative_edges).T
    return neg_edge_tensor


def create_transductive_splits(
    dataset: BipartiteDataset,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Create transductive splits for bipartite graphs.
    
    All nodes are seen during training, only edges are split.
    
    Args:
        dataset: Original bipartite dataset
        train_ratio: Ratio of edges for training
        val_ratio: Ratio of edges for validation
        test_ratio: Ratio of edges for testing
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Shuffle edges
    num_edges = dataset.num_edges
    perm = torch.randperm(num_edges)
    
    # Split edges
    num_train = int(num_edges * train_ratio)
    num_val = int(num_edges * val_ratio)
    
    train_indices = perm[:num_train]
    val_indices = perm[num_train:num_train + num_val]
    test_indices = perm[num_train + num_val:]
    
    # Create negative edges
    train_neg_edges = _sample_negative_edges(dataset, train_indices, len(train_indices))
    val_neg_edges = _sample_negative_edges(dataset, val_indices, len(val_indices))
    test_neg_edges = _sample_negative_edges(dataset, test_indices, len(test_indices))
    
    splits = {}
    
    for split_name, indices, neg_edges in [
        ('train', train_indices, train_neg_edges),
        ('val', val_indices, val_neg_edges),
        ('test', test_indices, test_neg_edges)
    ]:
        splits[split_name] = {
            'pos_edge_index': dataset.edge_index[:, indices],
            'neg_edge_index': neg_edges,
            'x_u': dataset.x_u,
            'x_v': dataset.x_v,
            'num_nodes_u': dataset.num_nodes_u,
            'num_nodes_v': dataset.num_nodes_v,
            'node_map_u': torch.arange(dataset.num_nodes_u),
            'node_map_v': torch.arange(dataset.num_nodes_v)
        }
    
    return splits
