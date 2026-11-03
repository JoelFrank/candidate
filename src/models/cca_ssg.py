"""
CCA-SSG (Canonical Correlation Analysis - Self-Supervised Graphs) for bipartite graphs.

CCA-based approach adapted for bipartite graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .base import BipartiteGNN, BipartiteAugmentation


class CCASSG(nn.Module):
    """
    CCA-SSG model for bipartite graphs.
    
    Uses Canonical Correlation Analysis loss for self-supervised learning.
    """
    
    def __init__(
        self,
        input_dim_u: int,
        input_dim_v: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        lambda_param: float = 0.005,  # Regularization parameter
        conv_type: str = 'gcn',
        num_v_attrs: int = 0,  # Number of V node attributes
        v_attr_dim: int = 64   # Dimension for V attribute embeddings
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lambda_param = lambda_param
        
        # Shared encoder for both augmentations (same as SNAP)
        self.encoder = BipartiteGNN(
            input_dim_u, input_dim_v, hidden_dim, output_dim,
            num_layers, dropout, conv_type, num_v_attrs=num_v_attrs, v_attr_dim=v_attr_dim
        )
        
        # Note: SNAP original does NOT use separate projectors for CCA-SSG
        # It only uses the encoder directly with z-normalization
        
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
            edge_index_aug = self.augmentation.edge_dropout(edge_index, p=0.15)
            x_u_aug = x_u.clone()
            x_v_aug = x_v.clone()
        elif aug_type == 'feat_drop':
            edge_index_aug = edge_index.clone()
            x_u_aug = self.augmentation.feature_dropout(x_u, p=0.15)
            x_v_aug = self.augmentation.feature_dropout(x_v, p=0.15)
        elif aug_type == 'add_edges':
            num_nodes_v = x_v.size(0)
            edge_index_aug = self.augmentation.add_random_edges(
                edge_index, num_nodes_u, num_nodes_v, p=0.05
            )
            x_u_aug = x_u.clone()
            x_v_aug = x_v.clone()
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
        Forward pass for CCA-SSG training.
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u]
            x_v: Features for second node type [num_v, input_dim_v]
            edge_index: Edge indices [2, num_edges]
            num_nodes_u: Number of nodes of first type
            v_attr_ids: Optional attribute IDs for V nodes [num_v]
            
        Returns:
            Dictionary containing projections for both augmentations
        """
        # Ensure tensors are detached to avoid computational graph reuse
        x_u = x_u.clone()
        x_v = x_v.clone()
        edge_index = edge_index.clone().detach()
        
        # Apply two different augmentations
        x_u_aug1, x_v_aug1, edge_index_aug1 = self._apply_augmentation(
            x_u, x_v, edge_index, num_nodes_u, 'edge_drop'
        )
        x_u_aug2, x_v_aug2, edge_index_aug2 = self._apply_augmentation(
            x_u, x_v, edge_index, num_nodes_u, 'feat_drop'
        )
        
        # Encode both augmentations (no projectors, following SNAP)
        h_u_1, h_v_1 = self.encoder(x_u_aug1, x_v_aug1, edge_index_aug1, num_nodes_u, v_attr_ids=v_attr_ids)
        h_u_2, h_v_2 = self.encoder(x_u_aug2, x_v_aug2, edge_index_aug2, num_nodes_u, v_attr_ids=v_attr_ids)
        
        # Directly use encoder outputs (z-normalization in loss)
        return {
            'z_u_1': h_u_1,
            'z_v_1': h_v_1,
            'z_u_2': h_u_2,
            'z_v_2': h_v_2
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute CCA loss using canonical correlation analysis.
        Based on SNAP research implementation.
        
        Args:
            outputs: Dictionary from forward pass
            
        Returns:
            Total loss
        """
        z_u_1 = outputs['z_u_1']
        z_v_1 = outputs['z_v_1']
        z_u_2 = outputs['z_u_2']
        z_v_2 = outputs['z_v_2']
        
        # Concatenate both node types for unified CCA loss
        z1 = torch.cat([z_u_1, z_v_1], dim=0)
        z2 = torch.cat([z_u_2, z_v_2], dim=0)
        
        # Apply z-normalization (SNAP original)
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        
        # Compute CCA loss following SNAP formula
        loss = self._cca_loss_snap(z1, z2)
        
        return loss
    
    def _cca_loss_snap(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        CCA-SSG loss following exact SNAP implementation:
        https://github.com/snap-research/non-contrastive-link-prediction
        
        From src/lib/loss.py:cca_ssg_loss()
        
        Args:
            z1: First z-normalized representation [N, D]
            z2: Second z-normalized representation [N, D]
            
        Returns:
            CCA loss
        """
        N = z1.size(0)
        D = z1.size(1)
        
        # Cross-correlation matrix
        c = torch.mm(z1.T, z2) / N
        
        # Auto-correlation matrices
        c1 = torch.mm(z1.T, z1) / N
        c2 = torch.mm(z2.T, z2) / N
        
        # Loss components
        loss_inv = -torch.diagonal(c).sum()  # Maximize cross-correlation diagonal
        
        iden = torch.eye(D, device=z1.device)
        loss_dec1 = (iden - c1).pow(2).sum()  # c1 should be identity
        loss_dec2 = (iden - c2).pow(2).sum()  # c2 should be identity
        
        # Combined loss with lambda weighting
        loss = (1 - self.lambda_param) * loss_inv + self.lambda_param * (loss_dec1 + loss_dec2)
        
        return loss
    
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
