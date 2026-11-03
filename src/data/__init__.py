"""
Data loading and preprocessing utilities for bipartite graphs.
"""

from .bipartite_loader import BipartiteDataLoader
from .dataset import BipartiteDataset
from .splits import create_inductive_splits

__all__ = ['BipartiteDataLoader', 'BipartiteDataset', 'create_inductive_splits']
