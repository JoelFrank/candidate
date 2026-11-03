"""
Base class for bipartite graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from .graph_contract import undirected_bipartite
from typing import Tuple, Optional, Union
import numpy as np


class BipartiteGNN(nn.Module):
    """
    Base class for bipartite graph neural networks.
    
    In a bipartite graph, we have two types of nodes (e.g., users and items).
    This class handles the message passing between different node types.
    
    Supports two modes:
    1. Feature-based: nodes have explicit features (input_dim_u/v > 0)
    2. Embedding-based: nodes use learnable embeddings (use_node_embeddings=True)
    """
    
    def __init__(
        self,
        input_dim_u: int,
        input_dim_v: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: str = 'gcn',
        use_node_embeddings: bool = False,
        num_nodes_u: Optional[int] = None,
        num_nodes_v: Optional[int] = None,
        embedding_dim: int = 128,
        num_v_attrs: Optional[int] = None,  # ✅ NEW: Number of V node attribute categories (e.g., TIPO)
        v_attr_dim: int = 64  # ✅ NEW: Dimension for V attribute embeddings
    ):
        super().__init__()
        
        self.input_dim_u = input_dim_u  # First node type dimension
        self.input_dim_v = input_dim_v  # Second node type dimension
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_node_embeddings = use_node_embeddings
        self.num_v_attrs = num_v_attrs
        
        # Node embedding layers (for structural learning without features)
        if use_node_embeddings:
            assert num_nodes_u is not None and num_nodes_v is not None, \
                "Must provide num_nodes_u and num_nodes_v when use_node_embeddings=True"
            self.embedding_u = nn.Embedding(num_nodes_u, embedding_dim)
            
            # ✅ If V nodes have attributes (e.g., TIPO), concatenate attribute embedding with node embedding
            if num_v_attrs is not None and num_v_attrs > 0:
                self.embedding_v = nn.Embedding(num_nodes_v, embedding_dim - v_attr_dim)  # Reduce V embedding
                self.embedding_v_attr = nn.Embedding(num_v_attrs, v_attr_dim)  # Attribute embedding
                # Projection after concatenation
                self.proj_v = nn.Linear(embedding_dim, hidden_dim)  # embedding_dim = (embedding_dim - v_attr_dim) + v_attr_dim
            else:
                self.embedding_v = nn.Embedding(num_nodes_v, embedding_dim)
                self.proj_v = nn.Linear(embedding_dim, hidden_dim)
            
            # Project embeddings to hidden dimension
            self.proj_u = nn.Linear(embedding_dim, hidden_dim)
        else:
            # Input projections for different node types
            self.proj_u = nn.Linear(input_dim_u, hidden_dim)
            self.proj_v = nn.Linear(input_dim_v, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        conv_class = GCNConv if conv_type == 'gcn' else SAGEConv
        
        for i in range(num_layers):
            self.convs.append(conv_class(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_node_embeddings:
            nn.init.xavier_uniform_(self.embedding_u.weight)
            nn.init.xavier_uniform_(self.embedding_v.weight)
            if hasattr(self, 'embedding_v_attr'):  # ✅ Initialize V attribute embedding
                nn.init.xavier_uniform_(self.embedding_v_attr.weight)
        nn.init.xavier_uniform_(self.proj_u.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(
        self, 
        x_u: Union[torch.Tensor, torch.LongTensor], 
        x_v: Union[torch.Tensor, torch.LongTensor], 
        edge_index: torch.Tensor,
        num_nodes_u: int,
        edge_weight: Optional[torch.Tensor] = None,
        v_attr_ids: Optional[torch.LongTensor] = None  # ✅ NEW: V node attributes (e.g., TIPO IDs)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for bipartite graph.
        
        Args:
            x_u: Features or IDs for first node type 
                 [num_u, input_dim_u] if features
                 [num_u] if IDs (when use_node_embeddings=True)
            x_v: Features or IDs for second node type
                 [num_v, input_dim_v] if features
                 [num_v] if IDs (when use_node_embeddings=True)
            edge_index: Edge indices [2, num_edges]
            num_nodes_u: Number of nodes of first type
            edge_weight: Optional edge weights [num_edges] (for weighted GCN)
            v_attr_ids: Optional V node attribute IDs [num_v] (e.g., TIPO for each PATRIMÔNIO)
            
        Returns:
            Tuple of embeddings for both node types
        """
        # Get initial node features
        if self.use_node_embeddings:
            # x_u and x_v are node IDs
            h_u = self.proj_u(self.embedding_u(x_u))
            
            # ✅ If V nodes have attributes, concatenate attribute embedding with node embedding
            if v_attr_ids is not None and hasattr(self, 'embedding_v_attr'):
                h_v_node = self.embedding_v(x_v)  # [num_v, embedding_dim - v_attr_dim]
                h_v_attr = self.embedding_v_attr(v_attr_ids)  # [num_v, v_attr_dim]
                h_v_combined = torch.cat([h_v_node, h_v_attr], dim=-1)  # [num_v, embedding_dim]
                h_v = self.proj_v(h_v_combined)
            else:
                h_v = self.proj_v(self.embedding_v(x_v))
        else:
            # x_u and x_v are node features
            h_u = self.proj_u(x_u)
            h_v = self.proj_v(x_v)
        
        # Concatenate node features
        h = torch.cat([h_u, h_v], dim=0)  # [num_u + num_v, hidden_dim]
        
        edge_index, edge_weight = undirected_bipartite(edge_index, edge_weight, num_nodes_u)

        # Apply graph convolutions with optional edge weights
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight) if isinstance(conv, GCNConv) else conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Split back into node types
        h_u_out = h[:num_nodes_u]
        h_v_out = h[num_nodes_u:]
        
        # Apply output projection
        h_u_out = self.output_proj(h_u_out)
        h_v_out = self.output_proj(h_v_out)
        
        return h_u_out, h_v_out
    
    def encode(self, x_u: Union[torch.Tensor, torch.LongTensor], 
               x_v: Union[torch.Tensor, torch.LongTensor], 
               edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode node features (alias for forward method for compatibility).
        
        Args:
            x_u: Features or IDs for U nodes 
                 [num_u, feat_dim_u] if features
                 [num_u] if IDs (when use_node_embeddings=True)
            x_v: Features or IDs for V nodes
                 [num_v, feat_dim_v] if features
                 [num_v] if IDs (when use_node_embeddings=True)
            edge_index: Bipartite edge indices [2, num_edges]
            
        Returns:
            Tuple of embeddings for both node types
        """
        if self.use_node_embeddings:
            num_nodes_u = x_u.size(0)
        else:
            num_nodes_u = x_u.size(0)
        return self.forward(x_u, x_v, edge_index, num_nodes_u)


class BipartiteAugmentation:
    """
    Data augmentation strategies for bipartite graphs.
    """
    
    @staticmethod
    def edge_dropout(edge_index: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Random edge dropout."""
        if p <= 0:
            return edge_index
        
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) > p
        return edge_index[:, mask]
    
    @staticmethod
    def feature_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """
        Random feature dropout.
        
        Handles both:
        - Float tensors (features): Apply dropout mask
        - Long tensors (embedding indices): Return unchanged (embeddings handle dropout internally)
        """
        if p <= 0:
            return x
        
        # If x is indices (Long), return unchanged
        # Embedding layers will handle dropout during training
        if x.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
            return x
        
        # For float features, apply dropout mask
        mask = torch.rand_like(x, dtype=torch.float) > p
        return x * mask.float()
    
    @staticmethod
    def add_random_edges(
        edge_index: torch.Tensor, 
        num_nodes_u: int, 
        num_nodes_v: int, 
        p: float = 0.05
    ) -> torch.Tensor:
        """Add random edges to bipartite graph."""
        if p <= 0:
            return edge_index
        
        num_add = int(edge_index.size(1) * p)
        if num_add == 0:
            return edge_index
        
        # Generate random bipartite edges on the same device
        device = edge_index.device
        u_nodes = torch.randint(0, num_nodes_u, (num_add,), device=device)
        v_nodes = torch.randint(num_nodes_u, num_nodes_u + num_nodes_v, (num_add,), device=device)
        
        new_edges = torch.stack([u_nodes, v_nodes], dim=0)
        
        return torch.cat([edge_index, new_edges], dim=1)


class BipartiteCorruption:
    """
    Corruption strategies for generating negative samples in bipartite graphs.
    """
    
    @staticmethod
    def shuffle_features_u(
        x_u: torch.Tensor, 
        x_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shuffle features of first node type."""
        perm = torch.randperm(x_u.size(0), device=x_u.device)
        return x_u[perm], x_v
    
    @staticmethod
    def shuffle_features_v(
        x_u: torch.Tensor, 
        x_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shuffle features of second node type."""
        perm = torch.randperm(x_v.size(0), device=x_v.device)
        return x_u, x_v[perm]
    
    @staticmethod
    def random_edges(
        num_nodes_u: int, 
        num_nodes_v: int, 
        num_edges: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Generate random bipartite edges."""
        u_nodes = torch.randint(0, num_nodes_u, (num_edges,), device=device)
        v_nodes = torch.randint(num_nodes_u, num_nodes_u + num_nodes_v, (num_edges,), device=device)
        return torch.stack([u_nodes, v_nodes], dim=0)
