"""
Base Agent class for all simulation entities.
"""

import pygame
import random
import math


class Agent:
    """
    Base class for all agents in the simulation.
    
    Provides common functionality for position, velocity, acceleration,
    and steering behaviors.
    """
    
    def __init__(self, x: float, y: float, max_speed: float, max_force: float, config: dict):
        """
        Initialize an agent.
        
        Args:
            x: Initial x position
            y: Initial y position
            max_speed: Maximum speed of the agent
            max_force: Maximum steering force
            config: Configuration dictionary
        """
        self.position = pygame.Vector2(x, y)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1.0, max_speed)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = max_speed * config.get("speedMultiplier", 1.0)
        self.max_force = max_force * config.get("accelerationFactor", 1.0)
        self.config = config
    
    def apply_force(self, force: pygame.Vector2) -> None:
        """
        Apply a force to the agent's acceleration.
        
        Args:
            force: Force vector to apply
        """
        self.acceleration += force
    
    def update(self) -> None:
        """Update the agent's position based on velocity and acceleration."""
        # Add small random noise to simulate natural movement variations
        if self.config.get("movementNoise", 0) > 0:
            noise = pygame.Vector2(
                random.uniform(-self.config["movementNoise"], self.config["movementNoise"]),
                random.uniform(-self.config["movementNoise"], self.config["movementNoise"])
            )
            self.apply_force(noise)
        
        self.velocity += self.acceleration
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        
        self.position += self.velocity
        self.acceleration *= 0
    
    def steering(self, desired: pygame.Vector2) -> pygame.Vector2:
        """
        Calculate steering force toward a desired velocity.
        
        Args:
            desired: The desired velocity vector
            
        Returns:
            Steering force vector
        """
        steer = desired - self.velocity
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

