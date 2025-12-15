"""
Hunting behavior module containing clustering and coordinated hunting logic.
"""

from .clustering import PreyCluster, detect_prey_clusters
from .pack import HuntingPack

__all__ = ['PreyCluster', 'detect_prey_clusters', 'HuntingPack']

