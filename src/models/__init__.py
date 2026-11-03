"""
Bipartite Graph Neural Network models for non-contrastive self-supervised learning.
"""

from .tbgrl import TBGRL
from .bgrl import BGRL
from .gbt import GBT
from .cca_ssg import CCASSG
from .wbtbgrl import WBTBGRL

__all__ = [
    'TBGRL', 'BGRL', 'GBT', 'CCASSG', 'WBTBGRL',
]
