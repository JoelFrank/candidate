"""
Decoder models for link prediction in bipartite graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class BipartiteDecoder(nn.Module):
    """
    Decoder for link prediction in bipartite graphs.
    
    Takes embeddings of U and V nodes and predicts edge probabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        final_activation: str = 'sigmoid'
    ):
        """
        Initialize bipartite decoder.
        
        Args:
            input_dim: Input dimension (2 * embedding_dim for concatenation)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
            final_activation: Final activation ('sigmoid', 'none')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Final activation
        if final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif final_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        # 'none' means no final activation
        
        self.mlp = nn.Sequential(*layers)
        
        self.reset_parameters()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def reset_parameters(self):
        """Reset parameters using Xavier initialization."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        edge_index: torch.Tensor,
        combination_method: str = 'hadamard'
    ) -> torch.Tensor:
        """
        Forward pass for link prediction.
        
        Args:
            embeddings_u: Embeddings for U nodes [num_u, dim]
            embeddings_v: Embeddings for V nodes [num_v, dim]
            edge_index: Edge indices [2, num_edges]
            combination_method: How to combine embeddings ('hadamard', 'concat', 'l1', 'l2')
            
        Returns:
            Link prediction scores [num_edges, output_dim]
        """
        # Get embeddings for edge endpoints
        u_embs = embeddings_u[edge_index[0]]  # [num_edges, dim]
        
        # Adjust V indices to local indexing (subtract offset)
        # edge_index[1] is in range [num_u, num_u + num_v - 1], need [0, num_v - 1]
        v_indices_local = edge_index[1] - embeddings_u.size(0)
        v_embs = embeddings_v[v_indices_local]  # [num_edges, dim]
        
        # Combine embeddings
        if combination_method == 'hadamard':
            combined = u_embs * v_embs
        elif combination_method == 'concat':
            combined = torch.cat([u_embs, v_embs], dim=1)
        elif combination_method == 'l1':
            combined = torch.abs(u_embs - v_embs)
        elif combination_method == 'l2':
            combined = (u_embs - v_embs) ** 2
        elif combination_method == 'avg':
            combined = (u_embs + v_embs) / 2
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        # Pass through MLP
        scores = self.mlp(combined)
        
        return scores.squeeze(-1) if self.output_dim == 1 else scores
    
    def predict_all_links(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        combination_method: str = 'hadamard',
        batch_size: int = 10000
    ) -> torch.Tensor:
        """
        Predict scores for all possible U-V pairs.
        
        Args:
            embeddings_u: Embeddings for U nodes [num_u, dim]
            embeddings_v: Embeddings for V nodes [num_v, dim]
            combination_method: How to combine embeddings
            batch_size: Batch size for processing (to avoid memory issues)
            
        Returns:
            Link prediction scores [num_u, num_v]
        """
        num_u = embeddings_u.size(0)
        num_v = embeddings_v.size(0)
        
        scores = torch.zeros(num_u, num_v, device=embeddings_u.device)
        
        # Process in batches to avoid memory issues
        for i in range(0, num_u, batch_size):
            end_i = min(i + batch_size, num_u)
            u_batch = embeddings_u[i:end_i]  # [batch_u, dim]
            
            for j in range(0, num_v, batch_size):
                end_j = min(j + batch_size, num_v)
                v_batch = embeddings_v[j:end_j]  # [batch_v, dim]
                
                # Create all pairs for this batch
                u_expanded = u_batch.unsqueeze(1).expand(-1, end_j - j, -1)  # [batch_u, batch_v, dim]
                v_expanded = v_batch.unsqueeze(0).expand(end_i - i, -1, -1)  # [batch_u, batch_v, dim]
                
                # Combine embeddings
                if combination_method == 'hadamard':
                    combined = u_expanded * v_expanded
                elif combination_method == 'concat':
                    combined = torch.cat([u_expanded, v_expanded], dim=-1)
                elif combination_method == 'l1':
                    combined = torch.abs(u_expanded - v_expanded)
                elif combination_method == 'l2':
                    combined = (u_expanded - v_expanded) ** 2
                elif combination_method == 'avg':
                    combined = (u_expanded + v_expanded) / 2
                else:
                    raise ValueError(f"Unknown combination method: {combination_method}")
                
                # Flatten for MLP input
                combined_flat = combined.view(-1, combined.size(-1))
                
                # Predict scores
                batch_scores = self.mlp(combined_flat)
                if self.output_dim == 1:
                    batch_scores = batch_scores.squeeze(-1)
                
                # Reshape back to batch format
                batch_scores = batch_scores.view(end_i - i, end_j - j)
                
                # Store in full scores matrix
                scores[i:end_i, j:end_j] = batch_scores
        
        return scores


class DotProductDecoder(nn.Module):
    """Simple dot product decoder for link prediction."""
    
    def __init__(self, normalize: bool = True):
        """
        Initialize dot product decoder.
        
        Args:
            normalize: Whether to normalize embeddings before dot product
        """
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using dot product.
        
        Args:
            embeddings_u: Embeddings for U nodes [num_u, dim]
            embeddings_v: Embeddings for V nodes [num_v, dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Link prediction scores [num_edges]
        """
        # Get embeddings for edge endpoints
        u_embs = embeddings_u[edge_index[0]]
        v_indices_local = edge_index[1] - embeddings_u.size(0)
        v_embs = embeddings_v[v_indices_local]
        
        # Normalize if requested
        if self.normalize:
            u_embs = F.normalize(u_embs, p=2, dim=1)
            v_embs = F.normalize(v_embs, p=2, dim=1)
        
        # Compute dot product
        scores = torch.sum(u_embs * v_embs, dim=1)
        
        return scores


class DistanceDecoder(nn.Module):
    """Distance-based decoder for link prediction."""
    
    def __init__(self, distance_type: str = 'l2', temperature: float = 1.0):
        """
        Initialize distance decoder.
        
        Args:
            distance_type: Type of distance ('l1', 'l2', 'cosine')
            temperature: Temperature parameter for softmax-like transformation
        """
        super().__init__()
        self.distance_type = distance_type
        self.temperature = temperature
    
    def forward(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using distance.
        
        Args:
            embeddings_u: Embeddings for U nodes [num_u, dim]
            embeddings_v: Embeddings for V nodes [num_v, dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Link prediction scores [num_edges] (higher = more likely)
        """
        # Get embeddings for edge endpoints
        u_embs = embeddings_u[edge_index[0]]
        v_indices_local = edge_index[1] - embeddings_u.size(0)
        v_embs = embeddings_v[v_indices_local]
        
        # Compute distance
        if self.distance_type == 'l1':
            distances = torch.norm(u_embs - v_embs, p=1, dim=1)
        elif self.distance_type == 'l2':
            distances = torch.norm(u_embs - v_embs, p=2, dim=1)
        elif self.distance_type == 'cosine':
            u_norm = F.normalize(u_embs, p=2, dim=1)
            v_norm = F.normalize(v_embs, p=2, dim=1)
            # Cosine distance = 1 - cosine similarity
            distances = 1 - torch.sum(u_norm * v_norm, dim=1)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        # Convert to similarity (higher = more similar)
        scores = torch.exp(-distances / self.temperature)
        
        return scores


class MLPDecoder(nn.Module):
    """
    MLP Decoder for link prediction training.
    Specifically designed for Phase 2 training where decoder is trained separately.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
        num_layers: int = 2  # Added for compatibility
    ):
        """
        Initialize MLP decoder.
        
        Args:
            input_dim: Input embedding dimension (concatenated, so actually input_dim * 2)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout probability
            num_layers: Number of layers (for compatibility, currently ignored)
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Takes concatenated embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for link prediction.
        
        Args:
            edge_embeddings: Concatenated edge embeddings [num_edges, input_dim]
            
        Returns:
            Link prediction logits [num_edges]
        """
        # Pass through MLP
        logits = self.mlp(edge_embeddings)
        
        return logits


# Alias for compatibility
LinkPredictor = MLPDecoder
