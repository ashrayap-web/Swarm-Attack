"""
Hunting pack coordination for cooperative predator behavior.
"""

import pygame
import math
from typing import List, Optional, Dict
from .clustering import PreyCluster, detect_prey_clusters


class HuntingPack:
    """
    Manages coordinated hunting behavior among multiple predators.
    
    Coordinates predator positions around prey groups, manages collective
    state transitions, and enables synchronized burst attacks.
    """
    
    def __init__(self, predators: List, config: dict):
        """
        Initialize a hunting pack.
        
        Args:
            predators: List of predator agents in the pack
            config: Configuration dictionary
        """
        self.predators = predators
        self.config = config
        self.collective_state = "herding"
        self.burst_start_time = None
        self.prey_centroid: Optional[pygame.Vector2] = None
        self.formation_quality = 0.0
        self.assigned_angles: Dict = {}
        self.target_cluster: Optional[PreyCluster] = None
        self.all_clusters: List[PreyCluster] = []
    
    def update_collective_state(self, boids: List, current_time: int) -> None:
        """
        Update the collective hunting state for all predators.
        
        Args:
            boids: List of boid agents
            current_time: Current simulation time in milliseconds
        """
        if not boids or not self.predators:
            return
        
        # Detect prey clusters using smart clustering
        if self.config.get("enableClustering", True):
            self.all_clusters = detect_prey_clusters(boids, self.config)
            self.target_cluster = self._select_target_cluster()
            
            if self.target_cluster:
                self.prey_centroid = self.target_cluster.centroid
            else:
                self.prey_centroid = self._calculate_global_centroid(boids)
        else:
            self.prey_centroid = self._calculate_global_centroid(boids)
            self.target_cluster = PreyCluster(boids)
            self.all_clusters = [self.target_cluster]
        
        # Assign formation positions
        self._assign_formation_positions()
        
        # Calculate formation quality
        self.formation_quality = self._calculate_formation_quality()
        
        # Calculate prey density
        if self.target_cluster:
            prey_density = self.target_cluster.size
        else:
            prey_density = self._calculate_prey_density_at_centroid(boids)
        
        # State transitions
        if self.collective_state == "herding":
            if (self.formation_quality >= self.config['formationQualityThreshold'] and 
                prey_density >= self.config['preyDensityThreshold']):
                self.collective_state = "burst"
                self.burst_start_time = current_time
                for pred in self.predators:
                    pred.state = "burst"
                    pred.max_speed = self.config['burstSpeed']
        
        elif self.collective_state == "burst":
            if self.burst_start_time and (current_time - self.burst_start_time) >= self.config['burstDuration']:
                self.collective_state = "herding"
                self.burst_start_time = None
                for pred in self.predators:
                    pred.state = "herding"
                    pred.max_speed = self.config['cruiseSpeed']
    
    def _calculate_global_centroid(self, boids: List) -> pygame.Vector2:
        """Calculate global centroid of all boids."""
        centroid = pygame.Vector2(0, 0)
        for boid in boids:
            centroid += boid.position
        return centroid / len(boids)
    
    def _select_target_cluster(self) -> Optional[PreyCluster]:
        """
        Select which cluster to target based on strategy.
        
        Returns:
            Selected target cluster or None
        """
        if not self.all_clusters:
            return None
        
        # Filter clusters within max distance of any predator
        valid_clusters = []
        max_dist = self.config.get("maxTargetDistance", 400)
        
        for cluster in self.all_clusters:
            for pred in self.predators:
                if cluster.distance_to(pred.position) < max_dist:
                    valid_clusters.append(cluster)
                    break
        
        if not valid_clusters:
            valid_clusters = self.all_clusters
        
        strategy = self.config.get("targetStrategy", "nearest")
        
        if strategy == "largest":
            return max(valid_clusters, key=lambda c: c.size)
        elif strategy == "densest":
            return max(valid_clusters, key=lambda c: c.density)
        else:  # "nearest"
            pack_center = pygame.Vector2(0, 0)
            for pred in self.predators:
                pack_center += pred.position
            pack_center /= len(self.predators)
            return min(valid_clusters, key=lambda c: c.distance_to(pack_center))
    
    def _assign_formation_positions(self) -> None:
        """Assign each predator a target angle around the prey centroid."""
        if not self.predators or not self.prey_centroid:
            return
        
        num_predators = len(self.predators)
        angle_spacing = (2 * math.pi) / num_predators
        
        # Sort predators by current angle for smooth transitions
        predators_with_angles = []
        for pred in self.predators:
            to_prey = self.prey_centroid - pred.position
            if to_prey.length() > 0.001:
                current_angle = math.atan2(to_prey.y, to_prey.x)
            else:
                current_angle = 0
            predators_with_angles.append((pred, current_angle))
        
        predators_with_angles.sort(key=lambda x: x[1])
        
        # Assign evenly spaced target angles
        for i, (pred, _) in enumerate(predators_with_angles):
            target_angle = i * angle_spacing
            self.assigned_angles[pred] = target_angle
    
    def _calculate_formation_quality(self) -> float:
        """
        Calculate how well predators are distributed around prey.
        
        Returns:
            Quality score from 0.0 (poor) to 1.0 (perfect)
        """
        if not self.predators or len(self.predators) < 2:
            return 1.0
        
        ideal_spacing = (2 * math.pi) / len(self.predators)
        
        # Calculate actual angles
        actual_angles = []
        for pred in self.predators:
            to_prey = self.prey_centroid - pred.position
            if to_prey.length() > 0.001:
                angle = math.atan2(to_prey.y, to_prey.x)
                actual_angles.append(angle)
        
        if not actual_angles:
            return 0.0
        
        actual_angles.sort()
        
        # Calculate spacing errors
        spacing_errors = []
        for i in range(len(actual_angles)):
            next_i = (i + 1) % len(actual_angles)
            spacing = actual_angles[next_i] - actual_angles[i]
            if spacing < 0:
                spacing += 2 * math.pi
            error = abs(spacing - ideal_spacing)
            spacing_errors.append(error)
        
        avg_error = sum(spacing_errors) / len(spacing_errors) if spacing_errors else 0
        quality = 1.0 - min(1.0, avg_error / math.pi)
        
        # Also check distance quality
        distance_quality = 0.0
        for pred in self.predators:
            dist = pred.position.distance_to(self.prey_centroid)
            ideal_dist = self.config['formationRadius']
            dist_error = abs(dist - ideal_dist) / ideal_dist
            distance_quality += max(0, 1.0 - dist_error)
        distance_quality /= len(self.predators)
        
        return (quality * 0.6 + distance_quality * 0.4)
    
    def _calculate_prey_density_at_centroid(self, boids: List) -> int:
        """Calculate number of prey near the centroid."""
        if not self.prey_centroid:
            return 0
        
        count = 0
        radius = self.config['predatorPerception']
        for boid in boids:
            dist = self.prey_centroid.distance_to(boid.position)
            if dist < radius:
                count += 1
        
        return count
    
    def get_formation_position(self, predator) -> Optional[pygame.Vector2]:
        """
        Get the target formation position for a specific predator.
        
        Args:
            predator: The predator agent
            
        Returns:
            Target position in formation or None
        """
        if predator not in self.assigned_angles or not self.prey_centroid:
            return None
        
        target_angle = self.assigned_angles[predator]
        formation_radius = self.config['formationRadius']
        
        target_x = self.prey_centroid.x + formation_radius * math.cos(target_angle)
        target_y = self.prey_centroid.y + formation_radius * math.sin(target_angle)
        
        return pygame.Vector2(target_x, target_y)

