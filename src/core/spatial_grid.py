"""
Spatial hash grid for efficient neighbor lookup in 2D space.
"""

from collections import defaultdict
from typing import List, Tuple, Any


class SpatialGrid:
    """
    Spatial hash grid for efficient neighbor lookup.
    
    Divides the simulation space into cells and provides O(1) lookup
    for agents within a given radius by only checking relevant cells.
    """
    
    def __init__(self, width: int, height: int, cell_size: int):
        """
        Initialize the spatial grid.
        
        Args:
            width: Width of the simulation area
            height: Height of the simulation area
            cell_size: Size of each grid cell
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
    
    def clear(self) -> None:
        """Clear all agents from the grid."""
        self.grid.clear()
    
    def _hash(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell coordinates.
        
        Args:
            x: X position in world coordinates
            y: Y position in world coordinates
            
        Returns:
            Tuple of (column, row) cell indices
        """
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (max(0, min(col, self.cols - 1)), max(0, min(row, self.rows - 1)))
    
    def insert(self, agent: Any) -> None:
        """
        Insert an agent into the grid based on its position.
        
        Args:
            agent: Agent object with a 'position' attribute (pygame.Vector2)
        """
        cell = self._hash(agent.position.x, agent.position.y)
        self.grid[cell].append(agent)
    
    def get_neighbors(self, position: Any, radius: float) -> List[Any]:
        """
        Get all agents within a given radius of a position.
        
        Args:
            position: Center position (pygame.Vector2 or similar with distance_to method)
            radius: Search radius
            
        Returns:
            List of agents within the specified radius
        """
        neighbors = []
        cell = self._hash(position.x, position.y)
        cells_to_check = self._get_adjacent_cells(cell)
        
        for c in cells_to_check:
            for agent in self.grid.get(c, []):
                dist = position.distance_to(agent.position)
                if dist < radius:
                    neighbors.append(agent)
        
        return neighbors
    
    def _get_adjacent_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get a cell and its 8 neighboring cells.
        
        Args:
            cell: The center cell as (column, row)
            
        Returns:
            List of cell coordinates to check
        """
        col, row = cell
        cells = []
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                nc, nr = col + dc, row + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    cells.append((nc, nr))
        return cells

