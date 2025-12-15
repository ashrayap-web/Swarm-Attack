"""
Core module containing configuration, spatial grid, and agent classes.
"""

from .config import SimulationConfig, DEFAULT_CONFIG, BENCHMARK_CONFIG
from .spatial_grid import SpatialGrid

__all__ = ['SimulationConfig', 'DEFAULT_CONFIG', 'BENCHMARK_CONFIG', 'SpatialGrid']

