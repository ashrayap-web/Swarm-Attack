"""
Boid (prey) agent class implementing flocking behavior.
"""

import pygame
from .base import Agent
from ..config import (
    SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT,
    BOUNDARY_WEIGHT, PREDATOR_AVOID_WEIGHT, BOUNDARY_MARGIN
)


class Boid(Agent):
    """
    A boid (prey) agent that exhibits flocking behavior.
    
    Implements Reynolds' boid rules:
    - Separation: Avoid crowding neighbors
    - Alignment: Steer toward average heading of neighbors
    - Cohesion: Steer toward average position of neighbors
    
    Also includes predator avoidance and adaptive behavior.
    """
    
    def __init__(self, x: float, y: float, config: dict):
        """
        Initialize a boid.
        
        Args:
            x: Initial x position
            y: Initial y position
            config: Configuration dictionary
        """
        super().__init__(x, y, config["maxSpeed"], config["maxForce"], config)
        self.stress_level = 0.0
        self.adaptive_weights = {
            "separation": SEPARATION_WEIGHT,
            "alignment": ALIGNMENT_WEIGHT,
            "cohesion": COHESION_WEIGHT
        }
    
    def flock(self, spatial_grid, predators: list) -> None:
        """
        Apply flocking behavior based on nearby boids and predators.
        
        Args:
            spatial_grid: SpatialGrid for efficient neighbor lookup
            predators: List of predator agents to avoid
        """
        neighbors = spatial_grid.get_neighbors(self.position, self.config["neighborRadius"])
        neighbors = [n for n in neighbors if isinstance(n, Boid) and n is not self]
        
        # Calculate flocking forces
        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        
        # Boundary and predator forces
        boundary = self.boundary_repulsion()
        pred_avoid = self.avoid_predators(predators)
        
        # Get weights (adaptive or fixed)
        if self.config.get("adaptiveBehaviorEnabled", False):
            self.adapt_behavior(neighbors, predators)
            sep_w = self.adaptive_weights["separation"]
            ali_w = self.adaptive_weights["alignment"]
            coh_w = self.adaptive_weights["cohesion"]
        else:
            sep_w = SEPARATION_WEIGHT
            ali_w = ALIGNMENT_WEIGHT
            coh_w = COHESION_WEIGHT
        
        # Apply forces
        self.apply_force(sep * sep_w)
        self.apply_force(ali * ali_w)
        self.apply_force(coh * coh_w)
        self.apply_force(boundary * BOUNDARY_WEIGHT)
        self.apply_force(pred_avoid * PREDATOR_AVOID_WEIGHT)
    
    def separation(self, neighbors: list) -> pygame.Vector2:
        """
        Calculate separation steering to avoid crowding neighbors.
        
        Args:
            neighbors: List of nearby boid neighbors
            
        Returns:
            Separation steering force
        """
        steering = pygame.Vector2(0, 0)
        total = 0
        
        for other in neighbors:
            dist = self.position.distance_to(other.position)
            if 0 < dist < self.config["desiredSeparation"]:
                diff = self.position - other.position
                diff /= dist  # Weight by distance
                steering += diff
                total += 1
        
        if total > 0:
            steering /= total
            if steering.length() > 0.001:
                steering.scale_to_length(self.max_speed)
                steering = self.steering(steering)
        return steering
    
    def alignment(self, neighbors: list) -> pygame.Vector2:
        """
        Calculate alignment steering toward average neighbor heading.
        
        Args:
            neighbors: List of nearby boid neighbors
            
        Returns:
            Alignment steering force
        """
        steering = pygame.Vector2(0, 0)
        total = 0
        
        for other in neighbors:
            steering += other.velocity
            total += 1
        
        if total > 0:
            steering /= total
            if steering.length() > 0.001:
                steering.scale_to_length(self.max_speed)
                steering = self.steering(steering)
        return steering
    
    def cohesion(self, neighbors: list) -> pygame.Vector2:
        """
        Calculate cohesion steering toward average neighbor position.
        
        Args:
            neighbors: List of nearby boid neighbors
            
        Returns:
            Cohesion steering force
        """
        steering = pygame.Vector2(0, 0)
        total = 0
        center = pygame.Vector2(0, 0)
        
        for other in neighbors:
            center += other.position
            total += 1
        
        if total > 0:
            center /= total
            desired = center - self.position
            if desired.length() > 0.001:
                desired.scale_to_length(self.max_speed)
                steering = self.steering(desired)
        return steering
    
    def boundary_repulsion(self) -> pygame.Vector2:
        """
        Calculate repulsion force from simulation boundaries.
        
        Returns:
            Boundary repulsion steering force
        """
        steering = pygame.Vector2(0, 0)
        width = self.config["screenWidth"]
        height = self.config["screenHeight"]
        
        if self.position.x < BOUNDARY_MARGIN:
            force = pygame.Vector2(1, 0)
            force *= (BOUNDARY_MARGIN - self.position.x) / BOUNDARY_MARGIN
            steering += force
        
        if self.position.x > width - BOUNDARY_MARGIN:
            force = pygame.Vector2(-1, 0)
            force *= (self.position.x - (width - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force
        
        if self.position.y < BOUNDARY_MARGIN:
            force = pygame.Vector2(0, 1)
            force *= (BOUNDARY_MARGIN - self.position.y) / BOUNDARY_MARGIN
            steering += force
        
        if self.position.y > height - BOUNDARY_MARGIN:
            force = pygame.Vector2(0, -1)
            force *= (self.position.y - (height - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force
        
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def avoid_predators(self, predators: list) -> pygame.Vector2:
        """
        Calculate avoidance steering away from nearby predators.
        
        Args:
            predators: List of predator agents
            
        Returns:
            Predator avoidance steering force
        """
        steering = pygame.Vector2(0, 0)
        total = 0
        
        for pred in predators:
            dist = self.position.distance_to(pred.position)
            if dist < self.config["predatorAvoidRadius"]:
                diff = self.position - pred.position
                diff /= (dist + 1)
                steering += diff
                total += 1
                self.stress_level = min(1.0, self.stress_level + 0.1)
        
        if total > 0:
            steering /= total
            if steering.length() > 0.001:
                steering.scale_to_length(self.max_speed * 1.5)  # Flee faster
                steering = self.steering(steering)
        else:
            self.stress_level = max(0.0, self.stress_level - 0.02)
        
        return steering
    
    def adapt_behavior(self, neighbors: list, predators: list) -> None:
        """
        Adapt behavior weights based on environmental conditions.
        
        Args:
            neighbors: List of nearby boid neighbors
            predators: List of predator agents
        """
        if not self.config.get("adaptiveBehaviorEnabled", False):
            return
        
        lr = self.config.get("learningRate", 0.05)
        
        # Under stress: increase separation, decrease cohesion
        if self.stress_level > 0.5:
            self.adaptive_weights["separation"] += lr * 0.5
            self.adaptive_weights["cohesion"] -= lr * 0.2
        else:
            # Relax back to normal
            self.adaptive_weights["separation"] += lr * (SEPARATION_WEIGHT - self.adaptive_weights["separation"])
            self.adaptive_weights["alignment"] += lr * (ALIGNMENT_WEIGHT - self.adaptive_weights["alignment"])
            self.adaptive_weights["cohesion"] += lr * (COHESION_WEIGHT - self.adaptive_weights["cohesion"])
        
        # Clamp weights
        for key in self.adaptive_weights:
            self.adaptive_weights[key] = max(0.1, min(3.0, self.adaptive_weights[key]))
    
    def draw(self, surface, debug_mode: int = 0) -> None:
        """
        Draw the boid on the given surface.
        
        Args:
            surface: Pygame surface to draw on
            debug_mode: 0=normal, 1=velocity, 2=stress colors
        """
        color = self.config["boidColor"]
        
        if debug_mode == 2:
            stress_color = int(255 * self.stress_level)
            color = (stress_color, color[1], color[2])
        
        pygame.draw.circle(surface, color, (int(self.position.x), int(self.position.y)), 4)
        
        if debug_mode >= 1 and self.velocity.length() > 0:
            end_pos = self.position + self.velocity.normalize() * 10
            pygame.draw.line(surface, color, self.position, end_pos, 1)

