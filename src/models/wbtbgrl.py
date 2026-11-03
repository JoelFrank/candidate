"""
WBT-BGRL (Weighted Bipartite Triplet-Bootstrapped Graph Representation Learning)

This model extends T-BGRL to work with weighted bipartite graphs, where edge weights
represent frequencies or importance of connections between nodes.

Key Features:
- Weighted message passing in GCN layers
- Weight-sensitive edge dropout during augmentation
- Four variants controlling weight usage in pretraining and link prediction:
  * (WP, WB): Weighted Pretraining, Weighted Bottleneck (link prediction)
  * (WP, NWB): Weighted Pretraining, Non-Weighted Bottleneck
  * (NWP, WB): Non-Weighted Pretraining, Weighted Bottleneck
  * (NWP, NWB): Non-Weighted Pretraining, Non-Weighted Bottleneck

Reference:
    Based on the WBT-BGRL paper, extending T-BGRL for weighted bipartite graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Literal, List
import numpy as np
from torch_geometric.nn import GCNConv
from .graph_contract import undirected_bipartite



class WeightedGCNEncoder(nn.Module):
    """
    GCN Encoder that supports weighted edges for bipartite graphs.
    
    The weighted propagation in layer l is:
    H^(l+1) = ReLU(Â_weighted * H^(l) * W^(l))
    
    where Â_weighted incorporates edge weights during message passing.
    """
    
    def __init__(
        self,
        input_dim_u: int,
        input_dim_v: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim_u = input_dim_u
        self.input_dim_v = input_dim_v
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projections for different node types
        self.proj_u = nn.Linear(input_dim_u, hidden_dim)
        self.proj_v = nn.Linear(input_dim_v, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.proj_u.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes_u: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with weighted edges.
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u]
            x_v: Features for second node type [num_v, input_dim_v]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            num_nodes_u: Number of nodes of first type
            
        Returns:
            Tuple of embeddings for both node types
        """
        # Get initial node features
        h_u = self.proj_u(x_u)
        h_v = self.proj_v(x_v)
        
        # Concatenate node features
        h = torch.cat([h_u, h_v], dim=0)  # [num_u + num_v, hidden_dim]
        
        edge_index, edge_weight = undirected_bipartite(edge_index, edge_weight, num_nodes_u)

        # Apply graph convolutions with edge weights
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
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


class WeightedBipartiteAugmentation:
    """
    Augmentation strategies for weighted bipartite graphs.
    
    Features:
    - Weight-sensitive edge dropout: edges with higher weights are more likely to be retained
    - Feature dropout: randomly masks node features
    """
    
    @staticmethod
    def weighted_edge_dropout(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weight-sensitive edge dropout.
        
        The probability of retaining an edge is proportional to its normalized weight.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            p: Base dropout probability
            
        Returns:
            Tuple of (filtered edge_index, filtered edge_weight)
        """
        if p <= 0 or edge_weight is None or edge_weight.numel() == 0:
            return edge_index, edge_weight
        
        # Normalize weights to [0, 1]
        w_min = edge_weight.min()
        w_max = edge_weight.max()
        if w_max > w_min:
            w_norm = (edge_weight - w_min) / (w_max - w_min)
        else:
            w_norm = torch.zeros_like(edge_weight)
        
        # Probability of keeping an edge is: (1 - p) + p * w_norm
        # This means: higher weight -> higher retention probability
        keep_prob = (1 - p) + p * w_norm
        mask = torch.rand(edge_weight.size(0), device=edge_weight.device) < keep_prob
        
        return edge_index[:, mask], edge_weight[mask]
    
    @staticmethod
    def feature_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """
        Random feature dropout.
        
        Args:
            x: Node features [num_nodes, feat_dim]
            p: Dropout probability
            
        Returns:
            Masked features
        """
        if p <= 0:
            return x
        
        # If x is indices (Long), return unchanged
        if x.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
            return x
        
        # For float features, apply dropout mask
        mask = torch.rand_like(x, dtype=torch.float) > p
        return x * mask.float()


class WeightedBipartiteCorruption:
    """
    Corruption strategies for weighted bipartite graphs.
    
    Features:
    - Feature shuffling: randomly permutes node features
    - Random bipartite edges: generates random edges with minimum weight
    """
    
    @staticmethod
    def shuffle_features(
        x_u: torch.Tensor,
        x_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shuffle node features to create corrupted view."""
        # Check if using node IDs (Long) or features (Float)
        using_node_ids = x_u.dtype == torch.long
        
        if using_node_ids:
            # For node IDs, shuffle the IDs themselves
            perm_u = torch.randperm(x_u.size(0), device=x_u.device)
            perm_v = torch.randperm(x_v.size(0), device=x_v.device)
            return x_u[perm_u], x_v[perm_v]
        else:
            # For features, shuffle independently
            perm_u = torch.randperm(x_u.size(0), device=x_u.device)
            perm_v = torch.randperm(x_v.size(0), device=x_v.device)
            return x_u[perm_u], x_v[perm_v]
    
    @staticmethod
    def random_bipartite_edges(
        num_nodes_u: int,
        num_nodes_v: int,
        num_edges: int,
        min_weight: float = 1.0,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random bipartite edges with minimum weight.
        
        Args:
            num_nodes_u: Number of U nodes
            num_nodes_v: Number of V nodes
            num_edges: Number of random edges to generate
            min_weight: Minimum weight for corrupted edges
            device: Device to create tensors on
            
        Returns:
            Tuple of (edge_index, edge_weight)
        """
        u_nodes = torch.randint(0, num_nodes_u, (num_edges,), device=device)
        v_nodes = torch.randint(num_nodes_u, num_nodes_u + num_nodes_v, (num_edges,), device=device)
        edge_index = torch.stack([u_nodes, v_nodes], dim=0)
        edge_weight = torch.full((num_edges,), min_weight, device=device, dtype=torch.float)
        
        return edge_index, edge_weight


class WBTBGRL(nn.Module):
    """
    WBT-BGRL: Weighted Bipartite Triplet-Bootstrapped Graph Representation Learning.
    
    This model extends T-BGRL to incorporate edge weights representing frequencies
    or importance of connections in bipartite graphs.
    
    Four variants control weight usage:
    - use_weights_pretraining: Use weights in pretraining loss (WP vs NWP)
    - use_weights_link_prediction: Use weights in link prediction loss (WB vs NWB)
    
    Variants:
    - (WP, WB): Both losses use weights
    - (WP, NWB): Only pretraining uses weights
    - (NWP, WB): Only link prediction uses weights
    - (NWP, NWB): Neither loss uses weights
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
        lambda_param: float = 0.5,  # Balance between attractive and repulsive terms
        use_weights_pretraining: bool = True,  # WP vs NWP
        use_weights_link_prediction: bool = True,  # WB vs NWB
        feat_drop_rate: float = 0.1,
        edge_drop_rate: float = 0.1,
        min_corrupt_weight: float = 1.0,
        weight_propagation: Optional[bool] = None,
        weight_augmentation: Optional[bool] = None,
        weight_loss: Optional[bool] = None,
    ):
        """Initialise WBT-BGRL."""
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.lambda_param = lambda_param
        self.use_weights_pretraining = use_weights_pretraining
        self.use_weights_link_prediction = use_weights_link_prediction
        self.feat_drop_rate = feat_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.min_corrupt_weight = min_corrupt_weight
        self.weight_propagation = use_weights_pretraining if weight_propagation is None else weight_propagation
        self.weight_augmentation = use_weights_pretraining if weight_augmentation is None else weight_augmentation
        self.weight_loss = use_weights_pretraining if weight_loss is None else weight_loss

        # Online encoder (trainable)
        self.online_encoder = WeightedGCNEncoder(
            input_dim_u, input_dim_v, hidden_dim, output_dim,
            num_layers, dropout
        )
        
        # Target encoder (EMA updated)
        self.target_encoder = WeightedGCNEncoder(
            input_dim_u, input_dim_v, hidden_dim, output_dim,
            num_layers, dropout
        )
        
        # ✅ PAPER-COMPLIANT: Two independent predictors (one for each node type)
        # P_U = W_2^{P_U} · PReLU(W_1^{P_U} Z_U + b_1^{P_U}) + b_2^{P_U}
        self.predictor_u = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.PReLU(),  # Paper uses PReLU
            nn.Linear(hidden_dim, output_dim)
        )
        
        # P_V = W_2^{P_V} · PReLU(W_1^{P_V} Z_V + b_1^{P_V}) + b_2^{P_V}
        self.predictor_v = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.PReLU(),  # Paper uses PReLU
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize target encoder with online encoder parameters
        self._initialize_target_encoder()
        
        # Augmentation and corruption strategies
        self.augmentation = WeightedBipartiteAugmentation()
        self.corruption = WeightedBipartiteCorruption()
    
    def _initialize_target_encoder(self):
        """Initialize target encoder with online encoder parameters."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False
    
    def _update_target_encoder(self):
        """Update target encoder using exponential moving average (EMA)."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.tau * target_param.data +
                (1 - self.tau) * online_param.data
            )
    
    def _apply_augmentation_view1(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation for first view (edge dropout).
        
        Returns:
            Tuple of (x_u_aug, x_v_aug, edge_index_aug, edge_weight_aug)
        """
        # Edge dropout (weight-sensitive if weights are provided)
        if edge_weight is not None and self.weight_augmentation:
            edge_index_aug, edge_weight_aug = self.augmentation.weighted_edge_dropout(
                edge_index, edge_weight, self.edge_drop_rate
            )
        else:
            # Uniform edge dropout
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges, device=edge_index.device) > self.edge_drop_rate
            edge_index_aug = edge_index[:, mask]
            edge_weight_aug = edge_weight[mask] if edge_weight is not None else None
        
        return x_u, x_v, edge_index_aug, edge_weight_aug
    
    def _apply_augmentation_view2(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation for second view (feature dropout).
        
        Returns:
            Tuple of (x_u_aug, x_v_aug, edge_index_aug, edge_weight_aug)
        """
        # Feature dropout
        x_u_aug = self.augmentation.feature_dropout(x_u, self.feat_drop_rate)
        x_v_aug = self.augmentation.feature_dropout(x_v, self.feat_drop_rate)
        
        return x_u_aug, x_v_aug, edge_index, edge_weight
    
    def _apply_corruption(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: int,
        num_nodes_v: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply corruption to generate negative samples.
        
        Returns:
            Tuple of (x_u_corr, x_v_corr, edge_index_corr, edge_weight_corr)
        """
        # Shuffle features
        x_u_corr, x_v_corr = self.corruption.shuffle_features(x_u, x_v)
        
        # Generate random bipartite edges with minimum weight
        num_edges = edge_index.size(1)
        edge_index_corr, edge_weight_corr = self.corruption.random_bipartite_edges(
            num_nodes_u, num_nodes_v, num_edges,
            self.min_corrupt_weight, device=edge_index.device
        )
        
        return x_u_corr, x_v_corr, edge_index_corr, edge_weight_corr
    
    def forward(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: Optional[int] = None,
        edge_weight: Optional[torch.Tensor] = None,
        v_attr_ids: Optional[torch.Tensor] = None  # ✅ NEW: V node attributes
    ) -> Dict[str, Any]:
        """
        Forward pass for WBT-BGRL training.
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u] or node IDs
            x_v: Features for second node type [num_v, input_dim_v] or node IDs
            edge_index: Edge indices [2, num_edges]
            num_nodes_u: Number of nodes of first type (optional, inferred if not provided)
            edge_weight: Edge weights [num_edges] (optional)
            v_attr_ids: Optional attribute IDs for V nodes [num_v]  # ✅ NEW
            
        Returns:
            Dictionary containing embeddings, predictions, and edge information
        """
        # Infer num_nodes_u if not provided
        if num_nodes_u is None:
            num_nodes_u = x_u.size(0)
        
        num_nodes_v = x_v.size(0)
        
        # Generate augmented view 1 (edge dropout)
        x_u_aug1, x_v_aug1, edge_index_aug1, edge_weight_aug1 = \
            self._apply_augmentation_view1(x_u, x_v, edge_index, edge_weight)
        
        # Generate augmented view 2 (feature dropout)
        x_u_aug2, x_v_aug2, edge_index_aug2, edge_weight_aug2 = \
            self._apply_augmentation_view2(x_u, x_v, edge_index, edge_weight)
        
        # Generate corrupted view
        x_u_corr, x_v_corr, edge_index_corr, edge_weight_corr = \
            self._apply_corruption(x_u, x_v, edge_index, num_nodes_u, num_nodes_v)
        
        # Check if encoders support edge weights (WeightedGCNEncoder)
        encoder_supports_weights = hasattr(self.online_encoder, 'proj_u') and \
                                  not hasattr(self.online_encoder, 'use_node_embeddings')
        
        # Online encoder on first augmentation
        if encoder_supports_weights:
            h_u_online, h_v_online = self.online_encoder(
                x_u_aug1, x_v_aug1, edge_index_aug1, edge_weight_aug1 if self.weight_propagation else None, num_nodes_u
            )
        else:
            # Using standard BipartiteGNN (embedding-based)
            h_u_online, h_v_online = self.online_encoder(
                x_u_aug1, x_v_aug1, edge_index_aug1, num_nodes_u, v_attr_ids=v_attr_ids  # ✅ NEW
            )
        
        # ✅ PAPER-COMPLIANT: Apply INDEPENDENT predictors
        # P_U = predictor_u(h_U), P_V = predictor_v(h_V)
        z_u = self.predictor_u(h_u_online)
        z_v = self.predictor_v(h_v_online)
        
        # Target encoder on second augmentation
        with torch.no_grad():
            if encoder_supports_weights:
                h_u_target, h_v_target = self.target_encoder(
                    x_u_aug2, x_v_aug2, edge_index_aug2, edge_weight_aug2 if self.weight_propagation else None, num_nodes_u
                )
            else:
                h_u_target, h_v_target = self.target_encoder(
                    x_u_aug2, x_v_aug2, edge_index_aug2, num_nodes_u, v_attr_ids=v_attr_ids  # ✅ NEW
                )
        
        # Online encoder on corruption
        if encoder_supports_weights:
            h_u_corr, h_v_corr = self.online_encoder(
                x_u_corr, x_v_corr, edge_index_corr, edge_weight_corr if self.weight_propagation else None, num_nodes_u
            )
        else:
            h_u_corr, h_v_corr = self.online_encoder(
                x_u_corr, x_v_corr, edge_index_corr, num_nodes_u, v_attr_ids=v_attr_ids  # ✅ NEW
            )
        
        # ✅ PAPER-COMPLIANT: Apply INDEPENDENT predictors to corrupted views
        z_u_corr = self.predictor_u(h_u_corr)
        z_v_corr = self.predictor_v(h_v_corr)
        
        return {
            'z_u': z_u,
            'z_v': z_v,
            'h_u_target': h_u_target,
            'h_v_target': h_v_target,
            'z_u_corr': z_u_corr,
            'z_v_corr': z_v_corr,
            'edge_index_aug1': edge_index_aug1,
            'edge_weight_aug1': edge_weight_aug1,
            'edge_index_corr': edge_index_corr,
            'edge_weight_corr': edge_weight_corr
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, Any],
        num_nodes_u: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute weighted WBT-BGRL loss.
        
        L_WBT-BGRL = λ * L_repulsive + (1 - λ) * L_attractive
        
        where:
        - L_attractive: weighted average of cosine similarity over augmented edges
        - L_repulsive: weighted average of cosine similarity over corrupted edges (negative)
        
        Args:
            outputs: Dictionary from forward pass
            num_nodes_u: Number of nodes in first partition (optional, for weighted version)
            
        Returns:
            Total loss
        """
        z_u = outputs['z_u']
        z_v = outputs['z_v']
        h_u_target = outputs['h_u_target']
        h_v_target = outputs['h_v_target']
        z_u_corr = outputs['z_u_corr']
        z_v_corr = outputs['z_v_corr']
        
        # Check if we have edge information for weighted loss
        has_edge_info = ('edge_index_aug1' in outputs and 
                        'edge_weight_aug1' in outputs and
                        num_nodes_u is not None)
        
        if not has_edge_info:
            raise ValueError("Edge-wise SSL requires edge information and num_nodes_u.")
        # Preserve the manuscript's raw-weight edge objective; NWP changes only q to 1.
        attractive_loss = self._compute_weighted_edge_loss(
            z_u, h_v_target, outputs['edge_index_aug1'], outputs['edge_weight_aug1'],
            num_nodes_u, attractive=True)
        repulsive_loss = self._compute_weighted_edge_loss(
            z_u, z_v_corr, outputs['edge_index_corr'], outputs['edge_weight_corr'],
            num_nodes_u, attractive=False)

        # Combined loss
        total_loss = (
            (1 - self.lambda_param) * attractive_loss +
            self.lambda_param * repulsive_loss
        )
        
        return total_loss
    
    def _compute_cosine_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative cosine similarity loss (for node-level loss)."""
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        cosine_sim = torch.sum(z1_norm * z2_norm, dim=1)
        return -cosine_sim.mean()
    
    def _compute_weighted_edge_loss(
        self,
        z_u: torch.Tensor,
        h_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes_u: int,
        attractive: bool = True
    ) -> torch.Tensor:
        """
        Compute weighted edge-wise loss.
        
        For attractive term:
        L_attractive = -1/Σw * Σ w_uv * cos(pred(h_u^(1)), h_v^(2))
        
        For repulsive term:
        L_repulsive = 1/Σw * Σ w_uv * cos(pred(h_u^(1)), h_v^(c))
        
        Args:
            z_u: Predicted embeddings for U nodes [num_u, dim]
            h_v: Target embeddings for V nodes [num_v, dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            num_nodes_u: Number of U nodes
            attractive: If True, compute attractive loss; else repulsive
            
        Returns:
            Weighted loss
        """
        if edge_index.size(1) == 0:
            return (z_u.sum() + h_v.sum()) * 0.0
        if not bool(((edge_index[0] >= 0) & (edge_index[0] < num_nodes_u) &
                     (edge_index[1] >= num_nodes_u) &
                     (edge_index[1] < num_nodes_u + h_v.size(0))).all()):
            raise ValueError("Loss edges must be canonical U->V edges.")
        # Extract edge endpoints
        u_idx = edge_index[0]
        v_idx = edge_index[1] - num_nodes_u  # Adjust V indices to [0, num_v)
        
        # Get embeddings for edge endpoints
        z_u_edges = z_u[u_idx]  # [num_edges, dim]
        h_v_edges = h_v[v_idx]  # [num_edges, dim]
        
        # Normalize embeddings
        z_u_norm = F.normalize(z_u_edges, dim=1)
        h_v_norm = F.normalize(h_v_edges, dim=1)
        
        # Compute cosine similarity for each edge
        cos_sim = torch.sum(z_u_norm * h_v_norm, dim=1)  # [num_edges]
        
        # Apply weights if specified
        if self.weight_loss and edge_weight is not None:
            # Weighted average
            weight_sum = edge_weight.sum()
            if weight_sum > 0:
                weighted_cos = (edge_weight * cos_sim).sum() / weight_sum
            else:
                weighted_cos = cos_sim.mean()
        else:
            # Uniform average
            weighted_cos = cos_sim.mean()
        
        # Return loss (negative for attractive, positive for repulsive)
        if attractive:
            return -weighted_cos
        else:
            return weighted_cos
    
    def get_embeddings(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: Optional[int] = None,
        edge_weight: Optional[torch.Tensor] = None,
        v_attr_ids: Optional[torch.Tensor] = None  # ✅ NEW
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get node embeddings for evaluation.
        
        Args:
            x_u: Features for U nodes or node IDs
            x_v: Features for V nodes or node IDs
            edge_index: Edge indices
            num_nodes_u: Number of U nodes (optional, inferred if not provided)
            edge_weight: Edge weights (optional)
            v_attr_ids: Optional attribute IDs for V nodes [num_v]  # ✅ NEW
            
        Returns:
            Tuple of embeddings for both node types
        """
        if num_nodes_u is None:
            num_nodes_u = x_u.size(0)
        
        self.eval()
        with torch.no_grad():
            # Check if encoder supports weights
            encoder_supports_weights = hasattr(self.online_encoder, 'proj_u') and \
                                      not hasattr(self.online_encoder, 'use_node_embeddings')
            
            if encoder_supports_weights:
                h_u, h_v = self.online_encoder(x_u, x_v, edge_index, edge_weight if self.weight_propagation else None, num_nodes_u)
            else:
                # Using standard BipartiteGNN (embedding-based)
                h_u, h_v = self.online_encoder(x_u, x_v, edge_index, num_nodes_u, v_attr_ids=v_attr_ids)  # ✅ NEW
        
        return h_u, h_v
    
    def encode(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        v_attr_ids: Optional[torch.Tensor] = None  # ✅ NEW
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode node features (alias for get_embeddings for compatibility).
        
        Args:
            x_u: Features for first node type [num_u, input_dim_u] or node IDs
            x_v: Features for second node type [num_v, input_dim_v] or node IDs
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            v_attr_ids: Optional attribute IDs for V nodes [num_v]  # ✅ NEW
            
        Returns:
            Tuple of embeddings for both node types
        """
        num_nodes_u = x_u.size(0)
        return self.get_embeddings(x_u, x_v, edge_index, num_nodes_u, edge_weight, v_attr_ids)  # ✅ NEW
    
    def training_step(
        self,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_u: Optional[int] = None,
        edge_weight: Optional[torch.Tensor] = None,
        v_attr_ids: Optional[torch.Tensor] = None  # ✅ NEW
    ) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            x_u: Features for U nodes or node IDs
            x_v: Features for V nodes or node IDs
            edge_index: Edge indices
            num_nodes_u: Number of U nodes (optional, inferred if not provided)
            edge_weight: Edge weights (optional)
            v_attr_ids: Optional attribute IDs for V nodes [num_v]  # ✅ NEW
            
        Returns:
            Loss value
        """
        if num_nodes_u is None:
            num_nodes_u = x_u.size(0)
        
        self.train()
        
        # Forward pass
        outputs = self.forward(x_u, x_v, edge_index, num_nodes_u, edge_weight, v_attr_ids)  # ✅ NEW
        
        # Compute loss
        loss = self.compute_loss(outputs, num_nodes_u)
        
        # Caller must update EMA after optimizer.step(), as in revised/run.py.
        
        return loss
    
    def get_variant_name(self) -> str:
        """
        Get the name of the current variant based on configuration.
        
        Returns:
            Variant name string
        """
        wp = "WP" if self.use_weights_pretraining else "NWP"
        wb = "WB" if self.use_weights_link_prediction else "NWB"
        base = f"WBT-BGRL ({wp}, {wb})"
        return base

    @torch.no_grad()
    def build_temporal_embeddings(
        self,
        window_data: List[dict],
        feature_embedder=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode multiple temporal windows and return stacked embeddings.

        This method backs the Phase-2 temporal pipeline.  The encoder
        **must already be frozen** before calling it.

        Domain-agnostic: it accepts pre-built window descriptors so no
        dataset-specific logic lives inside the model.

        Parameters
        ----------
        window_data : list of dict
            One dict per temporal window, ordered earliest → latest
            (e.g. G_W1, G_W2, G_W3).  Each dict must contain:

            ``x_u``         FloatTensor [N_u_t, D]  — U-node features
            ``x_v``         FloatTensor [N_v_t, D]  — V-node features
            ``edge_index``  LongTensor  [2, E_t]
            ``edge_weight`` FloatTensor [E_t] or ``None``
            ``num_nodes_u`` int

        feature_embedder : nn.Module, optional
            Optional feature-embedding bridge (e.g. ``FeatureEmbedder`` or
            ``IndustryFeatureEmbedder``).  When provided, called as::

                x_u, x_v = feature_embedder(x_u, x_v)

            for each window before the graph encoder.  If ``None``,
            ``x_u`` / ``x_v`` are assumed to already be projected feature
            matrices matching the encoder's expected input dimension.

        Returns
        -------
        emb_stack_u : Tensor [N_u_total, T, output_dim]
        emb_stack_v : Tensor [N_v_total, T, output_dim]
        mask_u      : BoolTensor [N_u_total, T]
        mask_v      : BoolTensor [N_v_total, T]
            ``True`` where the node is present in that window, ``False``
            where absent (zero-padded slot, excluded from attention).

        Notes
        -----
        ``N_u_total`` / ``N_v_total`` are taken from the *last* window
        (largest cumulative graph G_W3).
        """
        if not window_data:
            raise ValueError("window_data must contain at least one window dict.")

        self.online_encoder.eval()
        if feature_embedder is not None:
            feature_embedder.eval()

        u_embs: List[torch.Tensor] = []
        v_embs: List[torch.Tensor] = []

        for w in window_data:
            x_u = w["x_u"]
            x_v = w["x_v"]
            edge_index = w["edge_index"]
            edge_weight = w.get("edge_weight", None)
            num_nodes_u = w["num_nodes_u"]

            if feature_embedder is not None:
                x_u, x_v = feature_embedder(x_u, x_v)

            h_u, h_v = self.online_encoder(
                x_u, x_v, edge_index, edge_weight, num_nodes_u
            )
            u_embs.append(h_u)   # [N_u_t, output_dim]
            v_embs.append(h_v)   # [N_v_t, output_dim]

        # Total node counts from the last (largest) window
        N_u_total = u_embs[-1].size(0)
        N_v_total = v_embs[-1].size(0)

        emb_stack_u, mask_u = stack_embeddings(u_embs, N_u_total)
        emb_stack_v, mask_v = stack_embeddings(v_embs, N_v_total)

        return emb_stack_u, emb_stack_v, mask_u, mask_v

    def apply_temporal_attention(
        self,
        emb_stack_u: torch.Tensor,
        emb_stack_v: torch.Tensor,
        mask_u: Optional[torch.Tensor] = None,
        mask_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply :class:`TemporalAttention` to stacked window embeddings.

        Called inside the Phase-2 training loop after the encoder is frozen.
        Gradients flow through ``self.temporal_attention``, so attention
        parameters are updated jointly with the decoder.

        Parameters
        ----------
        emb_stack_u : Tensor [N_u, T, output_dim]
        emb_stack_v : Tensor [N_v, T, output_dim]
        mask_u      : BoolTensor [N_u, T], optional
        mask_v      : BoolTensor [N_v, T], optional

        Returns
        -------
        h_u_final : Tensor [N_u, output_dim]
        h_v_final : Tensor [N_v, output_dim]

        Raises
        ------
        RuntimeError
            If ``temporal_mode != "attention"`` (module not in attention mode).
        """
        if self.temporal_mode != "attention" or self.temporal_attention is None:
            raise RuntimeError(
                "apply_temporal_attention() requires temporal_mode='attention'. "
                f"Current mode: '{self.temporal_mode}'."
            )

        h_u_final = self.temporal_attention(emb_stack_u, mask_u)  # [N_u, D]
        h_v_final = self.temporal_attention(emb_stack_v, mask_v)  # [N_v, D]
        return h_u_final, h_v_final
