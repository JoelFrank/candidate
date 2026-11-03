"""
BGRL (Bootstrapped Graph Representation Learning) for bipartite graphs.

Standard BGRL adapted for bipartite graph structure following SNAP implementation:
https://github.com/snap-research/non-contrastive-link-prediction

Key components from SNAP's BGRL:
- Online encoder + Target encoder (EMA updated)
- MLP predictor (2 layers with ReLU)
- Bootstrap latent loss: 2 - 2 * cosine_similarity
- No negative sampling required
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .base import BipartiteGNN, BipartiteAugmentation


class BGRL(nn.Module):
    """
    BGRL model for bipartite graphs.
    
    Standard bootstrapped learning approach without negative sampling,
    using only positive augmentations.
    """
    
    def __init__(
        self,
        input_dim_u: int,
        input_dim_v: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        tau: float = 0.995,  # EMA coefficient
        conv_type: str = 'gcn',
        num_v_attrs: int = 0,  # Number of V node attributes
        v_attr_dim: int = 64   # Dimension for V attribute embeddings
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        
        # Online encoder (trainable)
        self.online_encoder = BipartiteGNN(
            input_dim_u, input_dim_v, hidden_dim, output_dim,
            num_layers, dropout, conv_type, num_v_attrs=num_v_attrs, v_attr_dim=v_attr_dim
        )
        
        # Target encoder (EMA updated)
        self.target_encoder = BipartiteGNN(
            input_dim_u, input_dim_v, hidden_dim, output_dim,
            num_layers, dropout, conv_type, num_v_attrs=num_v_attrs, v_attr_dim=v_attr_dim
        )
        
        # Predictor (only for online encoder)
        # Following SNAP: 2-layer MLP with hidden_dim intermediate layer
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize target encoder with online encoder parameters
        self._initialize_target_encoder()
        
        # Augmentation strategies
        self.augmentation = BipartiteAugmentation()
    
    def _initialize_target_encoder(self):
        """Initialize target encoder with online encoder parameters."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False
    
    def _update_target_encoder(self):
        """Update target encoder using exponential moving average."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.tau * target_param.data + 
                (1 - self.tau) * online_param.data
            )
    
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
            edge_index_aug = self.augmentation.edge_dropout(edge_index, p=0.1)
            x_u_aug = x_u
            x_v_aug = x_v
        elif aug_type == 'feat_drop':
            edge_index_aug = edge_index
            x_u_aug = self.augmentation.feature_dropout(x_u, p=0.1)
            x_v_aug = self.augmentation.feature_dropout(x_v, p=0.1)
        elif aug_type == 'add_edges':
            num_nodes_v = x_v.size(0)
            edge_index_aug = self.augmentation.add_random_edges(
                edge_index, num_nodes_u, num_nodes_v, p=0.05
            )
            x_u_aug = x_u
            x_v_aug = x_v
        else:  # random combination
            edge_index_aug = self.augmentation.edge_dropout(edge_index, p=0.05)
            x_u_aug = self.augmentation.feature_dropout(x_u, p=0.05)
            x_v_aug = self.augmentation.feature_dropout(x_v, p=0.05)
        
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
        Forward pass for BGRL training.
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u]
            x_v: Features for second node type [num_v, input_dim_v]
            edge_index: Edge indices [2, num_edges]
            num_nodes_u: Number of nodes of first type
            v_attr_ids: Optional attribute IDs for V nodes [num_v]
            
        Returns:
            Dictionary containing embeddings and predictions
        """
        # Apply two different augmentations
        x_u_aug1, x_v_aug1, edge_index_aug1 = self._apply_augmentation(
            x_u, x_v, edge_index, num_nodes_u, 'edge_drop'
        )
        x_u_aug2, x_v_aug2, edge_index_aug2 = self._apply_augmentation(
            x_u, x_v, edge_index, num_nodes_u, 'feat_drop'
        )
        
        # Online encoder on first augmentation
        h_u_online, h_v_online = self.online_encoder(
            x_u_aug1, x_v_aug1, edge_index_aug1, num_nodes_u, v_attr_ids=v_attr_ids
        )
        
        # Apply predictor
        z_u = self.predictor(h_u_online)
        z_v = self.predictor(h_v_online)
        
        # Target encoder on second augmentation
        with torch.no_grad():
            h_u_target, h_v_target = self.target_encoder(
                x_u_aug2, x_v_aug2, edge_index_aug2, num_nodes_u, v_attr_ids=v_attr_ids
            )
        
        return {
            'z_u': z_u,
            'z_v': z_v,
            'h_u_target': h_u_target,
            'h_v_target': h_v_target
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute BGRL loss (only positive terms).
        
        Args:
            outputs: Dictionary from forward pass
            
        Returns:
            Total loss
        """
        z_u = outputs['z_u']
        z_v = outputs['z_v']
        h_u_target = outputs['h_u_target']
        h_v_target = outputs['h_v_target']
        
        # Positive terms only (attract augmented views)
        loss_u = self._compute_cosine_loss(z_u, h_u_target)
        loss_v = self._compute_cosine_loss(z_v, h_v_target)
        
        # Average loss across node types
        total_loss = (loss_u + loss_v) / 2
        
        return total_loss
    
    def _compute_cosine_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BGRL bootstrap latent loss.
        Following SNAP original: 2 - 2 * cosine_similarity
        """
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        cosine_sim = torch.sum(z1_norm * z2_norm, dim=1)
        return 2 - 2 * cosine_sim.mean()  # SNAP original formula
    
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
            h_u, h_v = self.online_encoder(x_u, x_v, edge_index, num_nodes_u, v_attr_ids=v_attr_ids)
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
        
        # Update target encoder
        self._update_target_encoder()
        
        return loss
