"""
Evaluators for link prediction in bipartite graphs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

try:
    from ..utils.metrics import compute_link_prediction_metrics, compute_edge_scores
except ImportError:
    # Fallback para cuando se ejecuta desde la raíz
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.metrics import compute_link_prediction_metrics, compute_edge_scores

from .decoder import BipartiteDecoder, DotProductDecoder


class LinkPredictionEvaluator:
    """
    Evaluator for link prediction tasks in bipartite graphs.
    """
    
    def __init__(
        self,
        decoder: Optional[nn.Module] = None,
        device: str = 'cpu',
        k_values: List[int] = [10, 50, 100]
    ):
        """
        Initialize evaluator.
        
        Args:
            decoder: Decoder model for link prediction
            device: Device to run evaluation on
            k_values: Values of k for Hits@k evaluation
        """
        self.decoder = decoder or DotProductDecoder()
        self.device = device
        self.k_values = k_values
        
        if decoder is not None:
            self.decoder.to(device)
    
    def evaluate_split(
        self,
        model: nn.Module,
        data_split: Dict[str, Any],
        split_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model on a data split.
        
        Args:
            model: Trained model
            data_split: Data split dictionary
            split_name: Name of the split for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Get embeddings
            x_u = data_split['x_u'].to(self.device)
            x_v = data_split['x_v'].to(self.device)
            pos_edges = data_split['pos_edge_index'].to(self.device)
            
            # Handle negative edges - generate if not present
            if 'neg_edge_index' in data_split:
                neg_edges = data_split['neg_edge_index'].to(self.device)
            else:
                # Generate negative edges for evaluation
                neg_edges = self._generate_negative_edges(
                    pos_edges, 
                    data_split['num_nodes_u'], 
                    data_split['num_nodes_v'],
                    num_neg_per_pos=1
                ).to(self.device)
            
            num_nodes_u = data_split['num_nodes_u']
            
            # Get node embeddings from model
            embeddings_u, embeddings_v = model.get_embeddings(
                x_u, x_v, pos_edges, num_nodes_u
            )
            
            # Compute scores
            pos_scores = self._compute_scores(embeddings_u, embeddings_v, pos_edges)
            neg_scores = self._compute_scores(embeddings_u, embeddings_v, neg_edges)
            
            # Compute metrics
            metrics = compute_link_prediction_metrics(
                pos_scores, neg_scores, self.k_values
            )
            
            # Add split name prefix
            prefixed_metrics = {f"{split_name}_{k}": v for k, v in metrics.items()}
            
        return prefixed_metrics
    
    def _compute_scores(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute link prediction scores."""
        if hasattr(self.decoder, 'forward'):
            # Use decoder's forward method
            scores = self.decoder(embeddings_u, embeddings_v, edge_index)
        else:
            # Use simple dot product
            scores = compute_edge_scores(
                embeddings_u, embeddings_v, edge_index, 'dot_product'
            )
        return scores
    
    def evaluate_all_splits(
        self,
        model: nn.Module,
        data_loader
    ) -> Dict[str, float]:
        """
        Evaluate model on all data splits.
        
        Args:
            model: Trained model
            data_loader: Data loader with train/val/test splits
            
        Returns:
            Dictionary of all evaluation metrics
        """
        all_metrics = {}
        
        # Evaluate on each split
        for split_name in ['train', 'val', 'test']:
            split_data = getattr(data_loader, f'get_{split_name}_data')()
            split_metrics = self.evaluate_split(model, split_data, split_name)
            all_metrics.update(split_metrics)
        
        return all_metrics
    
    def _generate_negative_edges(
        self, 
        pos_edges: torch.Tensor, 
        num_nodes_u: int, 
        num_nodes_v: int,
        num_neg_per_pos: int = 1
    ) -> torch.Tensor:
        """Generate negative edges that don't exist in positive edges."""
        # Convert positive edges to set for fast lookup
        pos_edge_set = set()
        for i in range(pos_edges.size(1)):
            u_node = pos_edges[0, i].item()
            v_node = pos_edges[1, i].item()
            pos_edge_set.add((u_node, v_node))
        
        # Generate negative edges
        num_neg = pos_edges.size(1) * num_neg_per_pos
        neg_edges = []
        
        attempts = 0
        max_attempts = num_neg * 10  # Prevent infinite loops
        
        while len(neg_edges) < num_neg and attempts < max_attempts:
            u_node = torch.randint(0, num_nodes_u, (1,)).item()
            v_node = torch.randint(num_nodes_u, num_nodes_u + num_nodes_v, (1,)).item()
            
            if (u_node, v_node) not in pos_edge_set:
                neg_edges.append([u_node, v_node])
            
            attempts += 1
        
        if len(neg_edges) < num_neg:
            # Fill remaining with random edges (even if duplicates)
            remaining = num_neg - len(neg_edges)
            u_nodes = torch.randint(0, num_nodes_u, (remaining,))
            v_nodes = torch.randint(num_nodes_u, num_nodes_u + num_nodes_v, (remaining,))
            for i in range(remaining):
                neg_edges.append([u_nodes[i].item(), v_nodes[i].item()])
        
        return torch.tensor(neg_edges[:num_neg], dtype=torch.long).T


class InductiveEvaluator(LinkPredictionEvaluator):
    """
    Specialized evaluator for inductive link prediction.
    
    Handles evaluation with unseen nodes and provides detailed analysis
    of performance on different edge types.
    """
    
    def __init__(
        self,
        decoder: Optional[nn.Module] = None,
        device: str = 'cpu',
        k_values: List[int] = [10, 50, 100]
    ):
        super().__init__(decoder, device, k_values)
    
    def evaluate_inductive_performance(
        self,
        model: nn.Module,
        test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Detailed evaluation for inductive setting.
        
        Args:
            model: Trained model
            test_data: Test data with observed/unobserved node information
            
        Returns:
            Dictionary with detailed metrics
        """
        model.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Get embeddings for all nodes
            x_u = test_data['x_u'].to(self.device)
            x_v = test_data['x_v'].to(self.device)
            pos_edges = test_data['pos_edge_index'].to(self.device)
            neg_edges = test_data['neg_edge_index'].to(self.device)
            num_nodes_u = test_data['num_nodes_u']
            
            embeddings_u, embeddings_v = model.get_embeddings(
                x_u, x_v, pos_edges, num_nodes_u
            )
            
            # Overall performance
            pos_scores = self._compute_scores(embeddings_u, embeddings_v, pos_edges)
            neg_scores = self._compute_scores(embeddings_u, embeddings_v, neg_edges)
            
            overall_metrics = compute_link_prediction_metrics(
                pos_scores, neg_scores, self.k_values
            )
            
            # Add prefix for overall metrics
            results = {f"test_{k}": v for k, v in overall_metrics.items()}
            
            # Analyze by edge type if observed/unobserved info is available
            if 'observed_nodes' in test_data and 'unobserved_nodes' in test_data:
                edge_type_metrics = self._analyze_by_edge_type(
                    embeddings_u, embeddings_v, pos_edges, neg_edges,
                    test_data['observed_nodes'], test_data['unobserved_nodes'],
                    num_nodes_u
                )
                results.update(edge_type_metrics)
        
        return results
    
    def _analyze_by_edge_type(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        observed_nodes: torch.Tensor,
        unobserved_nodes: torch.Tensor,
        num_nodes_u: int
    ) -> Dict[str, float]:
        """
        Analyze performance by edge type (observed-observed, observed-unobserved, etc.).
        """
        results = {}
        
        # Categorize edges by node types
        pos_categories = self._categorize_edges(pos_edges, observed_nodes, num_nodes_u)
        neg_categories = self._categorize_edges(neg_edges, observed_nodes, num_nodes_u)
        
        # Evaluate each category
        for category in ['obs_obs', 'obs_unobs', 'unobs_unobs']:
            if len(pos_categories[category]) > 0 and len(neg_categories[category]) > 0:
                pos_scores_cat = self._compute_scores(
                    embeddings_u, embeddings_v, pos_categories[category]
                )
                neg_scores_cat = self._compute_scores(
                    embeddings_u, embeddings_v, neg_categories[category]
                )
                
                cat_metrics = compute_link_prediction_metrics(
                    pos_scores_cat, neg_scores_cat, self.k_values
                )
                
                # Add category prefix
                for k, v in cat_metrics.items():
                    results[f"test_{category}_{k}"] = v
        
        return results
    
    def _categorize_edges(
        self,
        edges: torch.Tensor,
        observed_nodes: torch.Tensor,
        num_nodes_u: int
    ) -> Dict[str, torch.Tensor]:
        """
        Categorize edges based on whether endpoints are observed or unobserved.
        """
        categories = {'obs_obs': [], 'obs_unobs': [], 'unobs_unobs': []}
        
        observed_set = set(observed_nodes.cpu().numpy())
        
        for i in range(edges.size(1)):
            u_node = edges[0, i].item()
            v_node = edges[1, i].item()
            
            u_observed = u_node in observed_set
            v_observed = v_node in observed_set
            
            if u_observed and v_observed:
                categories['obs_obs'].append(i)
            elif u_observed or v_observed:
                categories['obs_unobs'].append(i)
            else:
                categories['unobs_unobs'].append(i)
        
        # Convert to tensors
        for category in categories:
            if len(categories[category]) > 0:
                indices = torch.tensor(categories[category], device=edges.device)
                categories[category] = edges[:, indices]
            else:
                categories[category] = torch.empty((2, 0), dtype=edges.dtype, device=edges.device)
        
        return categories
    
    def compute_generalization_gap(
        self,
        model: nn.Module,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        metric: str = 'hits@50'
    ) -> float:
        """
        Compute generalization gap between train and test performance.
        
        Args:
            model: Trained model
            train_data: Training data
            test_data: Test data
            metric: Metric to compare
            
        Returns:
            Generalization gap (train_metric - test_metric)
        """
        train_metrics = self.evaluate_split(model, train_data, 'train')
        test_metrics = self.evaluate_split(model, test_data, 'test')
        
        train_value = train_metrics.get(f'train_{metric}', 0.0)
        test_value = test_metrics.get(f'test_{metric}', 0.0)
        
        return train_value - test_value


class BatchedEvaluator(LinkPredictionEvaluator):
    """
    Evaluator that processes edges in batches to handle large graphs.
    """
    
    def __init__(
        self,
        decoder: Optional[nn.Module] = None,
        device: str = 'cpu',
        k_values: List[int] = [10, 50, 100],
        batch_size: int = 10000
    ):
        super().__init__(decoder, device, k_values)
        self.batch_size = batch_size
    
    def evaluate_split(
        self,
        model: nn.Module,
        data_split: Dict[str, Any],
        split_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model on a data split using batched processing.
        """
        model.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Get embeddings
            x_u = data_split['x_u'].to(self.device)
            x_v = data_split['x_v'].to(self.device)
            pos_edges = data_split['pos_edge_index'].to(self.device)
            neg_edges = data_split['neg_edge_index'].to(self.device)
            num_nodes_u = data_split['num_nodes_u']
            
            embeddings_u, embeddings_v = model.get_embeddings(
                x_u, x_v, pos_edges, num_nodes_u
            )
            
            # Compute scores in batches
            pos_scores = self._compute_scores_batched(
                embeddings_u, embeddings_v, pos_edges
            )
            neg_scores = self._compute_scores_batched(
                embeddings_u, embeddings_v, neg_edges
            )
            
            # Compute metrics
            metrics = compute_link_prediction_metrics(
                pos_scores, neg_scores, self.k_values
            )
            
            # Add split name prefix
            prefixed_metrics = {f"{split_name}_{k}": v for k, v in metrics.items()}
        
        return prefixed_metrics
    
    def _compute_scores_batched(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute scores in batches."""
        num_edges = edge_index.size(1)
        all_scores = []
        
        for i in range(0, num_edges, self.batch_size):
            end_i = min(i + self.batch_size, num_edges)
            batch_edges = edge_index[:, i:end_i]
            
            batch_scores = self._compute_scores(
                embeddings_u, embeddings_v, batch_edges
            )
            all_scores.append(batch_scores)
        
        return torch.cat(all_scores, dim=0)
