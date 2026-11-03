"""
GBT (Graph Barlow Twins) for bipartite graphs.

Barlow Twins adapted for bipartite graph structure following SNAP implementation:
https://github.com/snap-research/non-contrastive-link-prediction

Key components from SNAP's GraphBarlowTwins:
- Single shared encoder (no separate projector)
- Two augmented views of the graph
- Cross-correlation loss (diagonal -> 1, off-diagonal -> 0)
- Lambda parameter for off-diagonal penalty (default: 0.005)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .base import BipartiteGNN, BipartiteAugmentation


class GBT(nn.Module):
    """
    Graph Barlow Twins model for bipartite graphs.
    
    Uses cross-correlation loss to learn representations without negative sampling.
    """
    
    def __init__(
        self,
        input_dim_u: int,
        input_dim_v: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        lambda_param: float = 0.005,  # Weight for off-diagonal terms
        conv_type: str = 'gcn',
        num_v_attrs: int = 0,  # Number of V node attributes
        v_attr_dim: int = 64   # Dimension for V attribute embeddings
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lambda_param = lambda_param
        
        # Shared encoder for both augmentations
        self.encoder = BipartiteGNN(
            input_dim_u, input_dim_v, hidden_dim, output_dim,
            num_layers, dropout, conv_type, num_v_attrs=num_v_attrs, v_attr_dim=v_attr_dim
        )
        
        # Note: SNAP original GBT does NOT use a separate projector
        # It uses encoder outputs directly for Barlow Twins loss
        
        # Augmentation strategies
        self.augmentation = BipartiteAugmentation()
    
    def _apply_augmentation(
        self, 
        x_u: torch.Tensor, 
        x_v: torch.Tensor, 
        edge_index: torch.Tensor,
        num_nodes_u: int,
        aug_type: str = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply augmentation to bipartite graph."""
        if aug_type == 'edge_drop':
            edge_index_aug = self.augmentation.edge_dropout(edge_index, p=0.2)
            x_u_aug = x_u
            x_v_aug = x_v
        elif aug_type == 'feat_drop':
            edge_index_aug = edge_index
            x_u_aug = self.augmentation.feature_dropout(x_u, p=0.2)
            x_v_aug = self.augmentation.feature_dropout(x_v, p=0.2)
        elif aug_type == 'add_edges':
            num_nodes_v = x_v.size(0)
            edge_index_aug = self.augmentation.add_random_edges(
                edge_index, num_nodes_u, num_nodes_v, p=0.1
            )
            x_u_aug = x_u
            x_v_aug = x_v
        else:  # mixed augmentation
            edge_index_aug = self.augmentation.edge_dropout(edge_index, p=0.1)
            x_u_aug = self.augmentation.feature_dropout(x_u, p=0.1)
            x_v_aug = self.augmentation.feature_dropout(x_v, p=0.1)
        
        return x_u_aug, x_v_aug, edge_index_aug
    
    def forward(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: int,
        v_attr_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for GBT training.
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u]
            x_v: Features for second node type [num_v, input_dim_v]
            edge_index: Edge indices [2, num_edges]
            num_nodes_u: Number of nodes of first type
            v_attr_ids: Optional attribute IDs for V nodes [num_v]
            
        Returns:
            Dictionary containing projections for both augmentations
        """
        # Apply two different augmentations
        x_u_aug1, x_v_aug1, edge_index_aug1 = self._apply_augmentation(
            x_u, x_v, edge_index, num_nodes_u, 'edge_drop'
        )
        x_u_aug2, x_v_aug2, edge_index_aug2 = self._apply_augmentation(
            x_u, x_v, edge_index, num_nodes_u, 'feat_drop'
        )
        
        # Encode both augmentations
        h_u_1, h_v_1 = self.encoder(x_u_aug1, x_v_aug1, edge_index_aug1, num_nodes_u, v_attr_ids=v_attr_ids)
        h_u_2, h_v_2 = self.encoder(x_u_aug2, x_v_aug2, edge_index_aug2, num_nodes_u, v_attr_ids=v_attr_ids)
        
        # Use encoder outputs directly (SNAP original approach)
        # No separate projector as in SNAP's GraphBarlowTwins implementation
        return {
            'z_u_1': h_u_1,
            'z_v_1': h_v_1,
            'z_u_2': h_u_2,
            'z_v_2': h_v_2
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Barlow Twins loss using cross-correlation matrix.
        
        Args:
            outputs: Dictionary from forward pass
            
        Returns:
            Total loss
        """
        z_u_1 = outputs['z_u_1']
        z_v_1 = outputs['z_v_1']
        z_u_2 = outputs['z_u_2']
        z_v_2 = outputs['z_v_2']
        
        # Compute loss for each node type
        loss_u = self._barlow_twins_loss(z_u_1, z_u_2)
        loss_v = self._barlow_twins_loss(z_v_1, z_v_2)
        
        # Average across node types
        total_loss = (loss_u + loss_v) / 2
        
        return total_loss
    
    def _barlow_twins_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Barlow Twins loss for a pair of representations.
        
        Args:
            z1: First representation [batch_size, dim]
            z2: Second representation [batch_size, dim]
            
        Returns:
            Barlow Twins loss
        """
        # Normalize representations
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0, unbiased=False).clamp_min(1e-6)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0, unbiased=False).clamp_min(1e-6)
        
        # Compute cross-correlation matrix
        batch_size = z1.size(0)
        c = torch.mm(z1_norm.T, z2_norm) / batch_size
        
        # Compute loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        
        loss = on_diag + self.lambda_param * off_diag
        
        return loss
    
    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Extract off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def get_embeddings(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: int,
        v_attr_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get node embeddings for evaluation.
        
        Returns:
            Tuple of embeddings for both node types
        """
        self.eval()
        with torch.no_grad():
            h_u, h_v = self.encoder(x_u, x_v, edge_index, num_nodes_u, v_attr_ids=v_attr_ids)
        return h_u, h_v
    
    def encode(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        v_attr_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode node features (alias for get_embeddings for compatibility).
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u]
            x_v: Features for second node type [num_v, input_dim_v]
            edge_index: Edge indices [2, num_edges]
            v_attr_ids: Optional attribute IDs for V nodes [num_v]
            
        Returns:
            Tuple of embeddings for both node types
        """
        num_nodes_u = x_u.size(0)
        return self.get_embeddings(x_u, x_v, edge_index, num_nodes_u, v_attr_ids=v_attr_ids)
    
    def training_step(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: int,
        v_attr_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single training step.
        
        Returns:
            Loss value
        """
        self.train()
        
        # Forward pass
        outputs = self.forward(x_u, x_v, edge_index, num_nodes_u, v_attr_ids=v_attr_ids)
        
        # Compute loss
        loss = self.compute_loss(outputs)
        
        return loss
