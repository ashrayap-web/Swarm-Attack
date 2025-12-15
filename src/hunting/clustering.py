"""
Prey clustering algorithms for detecting fish groups.
"""

import pygame
import math
from typing import List, Optional


class PreyCluster:
    """
    Represents a cluster/group of prey fish.
    
    Used by predators to identify and target specific groups
    rather than the global centroid.
    """
    
    def __init__(self, fish_list: List):
        """
        Initialize a prey cluster.
        
        Args:
            fish_list: List of boid agents in this cluster
        """
        self.fish = fish_list
        self.centroid = self._calculate_centroid()
        self.size = len(fish_list)
        self.radius = self._calculate_radius()
        self.density = self._calculate_density()
    
    def _calculate_centroid(self) -> pygame.Vector2:
        """
        Calculate the center of mass of the cluster.
        
        Returns:
            Centroid position as Vector2
        """
        if not self.fish:
            return pygame.Vector2(0, 0)
        
        center = pygame.Vector2(0, 0)
        for fish in self.fish:
            center += fish.position
        return center / len(self.fish)
    
    def _calculate_radius(self) -> float:
        """
        Calculate the spread/radius of the cluster.
        
        Returns:
            Maximum distance from centroid to any fish
        """
        if not self.fish:
            return 0
        
        max_dist = 0
        for fish in self.fish:
            dist = self.centroid.distance_to(fish.position)
            if dist > max_dist:
                max_dist = dist
        return max_dist
    
    def _calculate_density(self) -> float:
        """
        Calculate density (fish per unit area).
        
        Returns:
            Fish count divided by circular area
        """
        if self.radius == 0:
            return float('inf') if self.size > 0 else 0
        area = math.pi * self.radius * self.radius
        return self.size / area if area > 0 else 0
    
    def distance_to(self, position: pygame.Vector2) -> float:
        """
        Distance from a position to this cluster's centroid.
        
        Args:
            position: Position to measure from
            
        Returns:
            Distance to centroid
        """
        return self.centroid.distance_to(position)
    
    def update(self) -> None:
        """Recalculate cluster properties."""
        self.centroid = self._calculate_centroid()
        self.size = len(self.fish)
        self.radius = self._calculate_radius()
        self.density = self._calculate_density()


def detect_prey_clusters(boids: List, config: dict) -> List[PreyCluster]:
    """
    Detect clusters of prey using distance-based clustering (similar to DBSCAN).
    
    This algorithm groups nearby boids into clusters, allowing predators to
    target specific groups rather than empty space between split groups.
    
    Args:
        boids: List of boid agents
        config: Configuration dictionary with clustering parameters
        
    Returns:
        List of PreyCluster objects sorted by size (largest first)
    """
    if not boids or not config.get("enableClustering", True):
        # Fallback: treat all boids as one cluster
        return [PreyCluster(boids)] if boids else []
    
    cluster_radius = config.get("clusterRadius", 60)
    min_cluster_size = config.get("minClusterSize", 5)
    
    # Track which boids have been assigned to clusters
    unassigned = set(boids)
    clusters = []
    
    while unassigned:
        # Start a new cluster with an arbitrary boid
        seed = unassigned.pop()
        current_cluster = [seed]
        
        # Expand cluster by finding neighbors within cluster_radius
        to_check = [seed]
        
        while to_check:
            checking_boid = to_check.pop(0)
            
            # Find all unassigned boids within cluster_radius
            neighbors = []
            for boid in list(unassigned):
                dist = checking_boid.position.distance_to(boid.position)
                if dist < cluster_radius:
                    neighbors.append(boid)
            
            # Add neighbors to cluster and queue for checking
            for neighbor in neighbors:
                if neighbor in unassigned:
                    current_cluster.append(neighbor)
                    unassigned.remove(neighbor)
                    to_check.append(neighbor)
        
        # Only keep clusters that meet minimum size
        if len(current_cluster) >= min_cluster_size:
            clusters.append(PreyCluster(current_cluster))
    
    # Sort by size (largest first)
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters

