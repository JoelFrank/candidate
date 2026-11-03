"""
Bipartite graph dataset implementation.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, List, Union
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler


class BipartiteDataset:
    """
    Dataset class for bipartite graphs.
    
    Handles bipartite graph structure with two node types and supports
    inductive splits for link prediction evaluation.
    """
    
    def __init__(
        self,
        edge_index: torch.Tensor,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        num_nodes_u: int,
        num_nodes_v: int,
        edge_labels: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        node_labels_u: Optional[torch.Tensor] = None,
        node_labels_v: Optional[torch.Tensor] = None
    ):
        """
        Initialize bipartite dataset.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            x_u: Features for first node type [num_u, feat_dim_u]
            x_v: Features for second node type [num_v, feat_dim_v]
            num_nodes_u: Number of nodes of first type
            num_nodes_v: Number of nodes of second type
            edge_labels: Optional edge labels
            edge_weight: Optional edge weights (e.g., frequencies) [num_edges]
            node_labels_u: Optional labels for first node type
            node_labels_v: Optional labels for second node type
        """
        self.edge_index = edge_index
        self.x_u = x_u
        self.x_v = x_v
        self.num_nodes_u = num_nodes_u
        self.num_nodes_v = num_nodes_v
        self.num_nodes = num_nodes_u + num_nodes_v
        self.num_edges = edge_index.size(1)
        
        self.edge_labels = edge_labels
        self.edge_weight = edge_weight
        self.node_labels_u = node_labels_u
        self.node_labels_v = node_labels_v
        
        # Validate bipartite structure
        self._validate_bipartite_structure()
    
    def _validate_bipartite_structure(self):
        """Validate that the graph is truly bipartite."""
        # Check that all edges connect nodes from different partitions
        sources = self.edge_index[0]
        targets = self.edge_index[1]
        
        # Sources should be in [0, num_nodes_u) and targets in [num_nodes_u, num_nodes_u + num_nodes_v)
        valid_sources = torch.all(sources < self.num_nodes_u)
        valid_targets = torch.all(targets >= self.num_nodes_u) and torch.all(targets < self.num_nodes)
        
        if not (valid_sources and valid_targets):
            raise ValueError("Edge index does not represent a valid bipartite graph")
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX bipartite graph."""
        G = nx.Graph()
        
        # Add nodes with bipartite labels
        for i in range(self.num_nodes_u):
            G.add_node(i, bipartite=0)
        for i in range(self.num_nodes_u, self.num_nodes):
            G.add_node(i, bipartite=1)
        
        # Add edges
        edges = self.edge_index.T.numpy()
        G.add_edges_from(edges)
        
        return G
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get adjacency matrix representation."""
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        adj[self.edge_index[0], self.edge_index[1]] = 1
        adj[self.edge_index[1], self.edge_index[0]] = 1  # Make symmetric
        return adj
    
    def get_edge_attr(self) -> Optional[torch.Tensor]:
        """Get edge attributes if available."""
        return self.edge_labels
    
    def compute_edge_weights_from_frequencies(self):
        """
        Compute edge weights based on edge frequencies.
        
        For edges that appear multiple times (duplicates), this counts their
        occurrences and stores them as weights. Removes duplicate edges.
        
        Returns:
            None (modifies self.edge_weight and potentially self.edge_index in-place)
        """
        # Convert to numpy for easier manipulation
        edges = self.edge_index.T.numpy()
        
        # Count edge frequencies
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        
        # Update edge_index with unique edges
        self.edge_index = torch.from_numpy(unique_edges.T).long()
        
        # Store counts as weights
        self.edge_weight = torch.from_numpy(counts).float()
        
        # Update number of edges
        self.num_edges = self.edge_index.size(1)
        
        print(f"Computed edge weights: {len(counts)} unique edges, "
              f"weight range [{counts.min()}, {counts.max()}]")
    
    def set_edge_weights(self, edge_weight: torch.Tensor):
        """
        Set edge weights manually.
        
        Args:
            edge_weight: Edge weights [num_edges]
        """
        if edge_weight.size(0) != self.num_edges:
            raise ValueError(f"Edge weight size {edge_weight.size(0)} doesn't match "
                           f"number of edges {self.num_edges}")
        self.edge_weight = edge_weight
    
    def get_edge_weights(self) -> Optional[torch.Tensor]:
        """
        Get edge weights.
        
        Returns:
            Edge weights if available, else None
        """
        return self.edge_weight
    
    def normalize_features(self, method: str = 'standard'):
        """
        Normalize node features.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'l2')
        """
        if method == 'standard':
            scaler_u = StandardScaler()
            scaler_v = StandardScaler()
            self.x_u = torch.tensor(
                scaler_u.fit_transform(self.x_u.numpy()), 
                dtype=torch.float
            )
            self.x_v = torch.tensor(
                scaler_v.fit_transform(self.x_v.numpy()), 
                dtype=torch.float
            )
        elif method == 'l2':
            self.x_u = torch.nn.functional.normalize(self.x_u, p=2, dim=1)
            self.x_v = torch.nn.functional.normalize(self.x_v, p=2, dim=1)
        elif method == 'minmax':
            self.x_u = (self.x_u - self.x_u.min(0)[0]) / (self.x_u.max(0)[0] - self.x_u.min(0)[0])
            self.x_v = (self.x_v - self.x_v.min(0)[0]) / (self.x_v.max(0)[0] - self.x_v.min(0)[0])
    
    def add_self_loops(self):
        """Add self-loops to the graph (not typically used for bipartite graphs)."""
        # For bipartite graphs, we don't add self-loops as they break the bipartite structure
        pass
    
    def remove_isolated_nodes(self):
        """Remove nodes with no connections."""
        # Get connected nodes
        connected_nodes = torch.unique(self.edge_index.flatten())
        
        # Find isolated nodes in each partition
        all_nodes_u = set(range(self.num_nodes_u))
        all_nodes_v = set(range(self.num_nodes_u, self.num_nodes))
        
        connected_u = set(connected_nodes[connected_nodes < self.num_nodes_u].tolist())
        connected_v = set(connected_nodes[connected_nodes >= self.num_nodes_u].tolist())
        
        isolated_u = all_nodes_u - connected_u
        isolated_v = all_nodes_v - connected_v
        
        if len(isolated_u) > 0 or len(isolated_v) > 0:
            print(f"Warning: Found {len(isolated_u)} isolated nodes in partition U and {len(isolated_v)} in partition V")
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get dataset statistics."""
        stats = {
            'num_nodes_u': self.num_nodes_u,
            'num_nodes_v': self.num_nodes_v,
            'num_nodes_total': self.num_nodes,
            'num_edges': self.num_edges,
            'density': self.num_edges / (self.num_nodes_u * self.num_nodes_v),
            'avg_degree_u': self.num_edges / self.num_nodes_u,
            'avg_degree_v': self.num_edges / self.num_nodes_v,
            'feature_dim_u': self.x_u.size(1),
            'feature_dim_v': self.x_v.size(1)
        }
        
        # Add edge weight statistics if available
        if self.edge_weight is not None:
            stats['has_edge_weights'] = True
            stats['edge_weight_min'] = float(self.edge_weight.min())
            stats['edge_weight_max'] = float(self.edge_weight.max())
            stats['edge_weight_mean'] = float(self.edge_weight.mean())
            stats['edge_weight_std'] = float(self.edge_weight.std())
        else:
            stats['has_edge_weights'] = False
        
        return stats
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"BipartiteDataset(nodes_u={stats['num_nodes_u']}, "
                f"nodes_v={stats['num_nodes_v']}, edges={stats['num_edges']}, "
                f"density={stats['density']:.4f})")


def load_synthetic_bipartite(
    num_nodes_u: int = 1000,
    num_nodes_v: int = 800,
    num_edges: int = 5000,
    feat_dim_u: int = 64,
    feat_dim_v: int = 32,
    seed: int = 42
) -> BipartiteDataset:
    """
    Generate synthetic bipartite graph for testing.
    
    Args:
        num_nodes_u: Number of nodes in first partition
        num_nodes_v: Number of nodes in second partition
        num_edges: Number of edges
        feat_dim_u: Feature dimension for first partition
        feat_dim_v: Feature dimension for second partition
        seed: Random seed
        
    Returns:
        BipartiteDataset instance
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random bipartite edges
    u_nodes = torch.randint(0, num_nodes_u, (num_edges,))
    v_nodes = torch.randint(num_nodes_u, num_nodes_u + num_nodes_v, (num_edges,))
    edge_index = torch.stack([u_nodes, v_nodes], dim=0)
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    # Generate random features
    x_u = torch.randn(num_nodes_u, feat_dim_u)
    x_v = torch.randn(num_nodes_v, feat_dim_v)
    
    return BipartiteDataset(
        edge_index=edge_index,
        x_u=x_u,
        x_v=x_v,
        num_nodes_u=num_nodes_u,
        num_nodes_v=num_nodes_v
    )
