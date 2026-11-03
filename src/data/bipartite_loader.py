"""
Bipartite graph data loader with support for inductive evaluation.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .dataset import BipartiteDataset, load_synthetic_bipartite
from .splits import create_inductive_splits, create_transductive_splits


class BipartiteDataLoader:
    """
    Data loader for bipartite graphs with support for both
    transductive and inductive evaluation settings.
    """
    
    def __init__(
        self,
        dataset: Optional[BipartiteDataset] = None,
        dataset_name: Optional[str] = None,
        evaluation_mode: str = 'inductive',
        split_params: Optional[Dict] = None
    ):
        """
        Initialize bipartite data loader.
        
        Args:
            dataset: BipartiteDataset instance
            dataset_name: Name of dataset to load ('synthetic' for synthetic data)
            evaluation_mode: 'inductive' or 'transductive'
            split_params: Parameters for data splitting
        """
        self.evaluation_mode = evaluation_mode
        self.split_params = split_params or {}
        
        # Load dataset
        if dataset is not None:
            self.dataset = dataset
        elif dataset_name == 'synthetic':
            self.dataset = load_synthetic_bipartite(**self.split_params.get('synthetic_params', {}))
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'synthetic'")
        
        # Create splits
        self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits based on evaluation mode."""
        if self.evaluation_mode == 'inductive':
            self.splits = create_inductive_splits(self.dataset, **self.split_params)
        else:  # transductive
            self.splits = create_transductive_splits(self.dataset, **self.split_params)
    
    def get_train_data(self) -> Dict:
        """Get training data."""
        return self.splits['train']
    
    def get_val_data(self) -> Dict:
        """Get validation data."""
        return self.splits['val']
    
    def get_test_data(self) -> Dict:
        """Get test data."""
        return self.splits['test']
    
    def get_full_data(self) -> BipartiteDataset:
        """Get full dataset."""
        return self.dataset
    
    def print_statistics(self):
        """Print dataset and split statistics."""
        print("=== Dataset Statistics ===")
        
        # Handle datasets with or without get_statistics method
        if hasattr(self.dataset, 'get_statistics'):
            stats = self.dataset.get_statistics()
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        else:
            # Basic statistics for datasets without get_statistics
            print(f"Total nodes U: {self.dataset.num_nodes_u}")
            print(f"Total nodes V: {self.dataset.num_nodes_v}")
            print(f"Total edges: {self.dataset.edge_index.size(1)}")
            print(f"U features: {self.dataset.x_u.shape}")
            print(f"V features: {self.dataset.x_v.shape}")
        
        print(f"\n=== Split Statistics ({self.evaluation_mode}) ===")
        for split_name in ['train', 'val', 'test']:
            split_data = self.splits[split_name]
            num_pos = split_data['pos_edge_index'].size(1)
            
            # Handle negative edges (may not exist for some datasets)
            if 'neg_edge_index' in split_data:
                num_neg = split_data['neg_edge_index'].size(1)
                neg_info = f", {num_neg} neg edges"
            else:
                neg_info = ""
            
            num_nodes_u = split_data['num_nodes_u']
            num_nodes_v = split_data['num_nodes_v']
            
            print(f"{split_name.upper()}: {num_pos} pos edges{neg_info}, "
                  f"{num_nodes_u} U nodes, {num_nodes_v} V nodes")


class BipartiteBatchDataset(Dataset):
    """
    PyTorch Dataset for batched training on bipartite graphs.
    """
    
    def __init__(
        self,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        x_u: torch.Tensor,
        x_v: torch.Tensor,
        num_nodes_u: int
    ):
        """
        Initialize batch dataset.
        
        Args:
            pos_edges: Positive edges [2, num_pos]
            neg_edges: Negative edges [2, num_neg]
            x_u: Features for first node type
            x_v: Features for second node type
            num_nodes_u: Number of nodes of first type
        """
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.x_u = x_u
        self.x_v = x_v
        self.num_nodes_u = num_nodes_u
        
        # Combine positive and negative edges
        self.all_edges = torch.cat([pos_edges, neg_edges], dim=1)
        self.labels = torch.cat([
            torch.ones(pos_edges.size(1)),
            torch.zeros(neg_edges.size(1))
        ])
    
    def __len__(self):
        return self.all_edges.size(1)
    
    def __getitem__(self, idx):
        edge = self.all_edges[:, idx]
        label = self.labels[idx]
        return edge, label


def create_link_prediction_batches(
    data_split: Dict,
    batch_size: int = 1024,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader for link prediction evaluation.
    
    Args:
        data_split: Data split dictionary
        batch_size: Batch size for evaluation
        shuffle: Whether to shuffle edges
        
    Returns:
        DataLoader for link prediction
    """
    dataset = BipartiteBatchDataset(
        pos_edges=data_split['pos_edge_index'],
        neg_edges=data_split['neg_edge_index'],
        x_u=data_split['x_u'],
        x_v=data_split['x_v'],
        num_nodes_u=data_split['num_nodes_u']
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to 0 for compatibility
    )


def load_real_bipartite_dataset(dataset_name: str) -> BipartiteDataset:
    """
    Load real-world bipartite datasets.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        BipartiteDataset instance
    """
    if dataset_name == 'movielens':
        return _load_movielens()
    elif dataset_name == 'amazon':
        return _load_amazon()
    elif dataset_name == 'goodreads':
        return _load_goodreads()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_movielens() -> BipartiteDataset:
    """Load MovieLens dataset as bipartite graph (users-movies)."""
    # This would load actual MovieLens data
    # For now, return synthetic data as placeholder
    print("Loading synthetic MovieLens-like data...")
    return load_synthetic_bipartite(
        num_nodes_u=1000,  # users
        num_nodes_v=1500,  # movies
        num_edges=8000,
        feat_dim_u=32,
        feat_dim_v=64
    )


def _load_amazon() -> BipartiteDataset:
    """Load Amazon dataset as bipartite graph (users-products)."""
    # This would load actual Amazon data
    # For now, return synthetic data as placeholder
    print("Loading synthetic Amazon-like data...")
    return load_synthetic_bipartite(
        num_nodes_u=2000,  # users
        num_nodes_v=3000,  # products
        num_edges=15000,
        feat_dim_u=64,
        feat_dim_v=128
    )


def _load_goodreads() -> BipartiteDataset:
    """Load Goodreads dataset as bipartite graph (users-books)."""
    # This would load actual Goodreads data
    # For now, return synthetic data as placeholder
    print("Loading synthetic Goodreads-like data...")
    return load_synthetic_bipartite(
        num_nodes_u=1500,  # users
        num_nodes_v=2500,  # books
        num_edges=12000,
        feat_dim_u=48,
        feat_dim_v=96
    )
