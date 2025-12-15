"""
Predator agent class implementing hunting behavior.
"""

import pygame
import math
import random
from .base import Agent
from ..config import BOUNDARY_MARGIN


class Predator(Agent):
    """
    A predator agent that hunts boids using herding and burst strategies.
    
    Supports both individual hunting and cooperative pack hunting.
    Uses predictive interception to catch prey efficiently.
    """
    
    def __init__(self, x: float, y: float, config: dict):
        """
        Initialize a predator.
        
        Args:
            x: Initial x position
            y: Initial y position
            config: Configuration dictionary
        """
        super().__init__(x, y, config["cruiseSpeed"], config["maxForce"] * 1.2, config)
        self.catch_flash = 0
        self.state = "herding"
        self.burst_start_time = None
        self.perception = config["predatorPerception"]
        
        # For benchmark tracking
        self.distance_traveled = 0.0
        self.previous_position = self.position.copy()
        
        # Store previous direction for realistic turning
        if self.velocity.length() > 0:
            self.previous_direction = math.atan2(self.velocity.y, self.velocity.x)
        else:
            self.previous_direction = 0.0
    
    def calculate_prey_centroid(self, boids: list) -> tuple:
        """
        Calculate the center of mass of prey within perception radius.
        
        Args:
            boids: List of boid agents
            
        Returns:
            Tuple of (centroid vector, prey count) or (None, 0)
        """
        center_of_mass = pygame.Vector2(0, 0)
        count = 0
        perception_sq = self.perception * self.perception
        
        for boid in boids:
            dist_sq = self.position.distance_squared_to(boid.position)
            if dist_sq < perception_sq:
                center_of_mass += boid.position
                count += 1
        
        if count > 0:
            return center_of_mass / count, count
        return None, 0
    
    def calculate_prey_density(self, boids: list) -> int:
        """
        Count prey within perception radius.
        
        Args:
            boids: List of boid agents
            
        Returns:
            Number of boids within perception radius
        """
        count = 0
        perception_sq = self.perception * self.perception
        
        for boid in boids:
            dist_sq = self.position.distance_squared_to(boid.position)
            if dist_sq < perception_sq:
                count += 1
        
        return count
    
    def herding_acceleration(self, prey_centroid: pygame.Vector2) -> pygame.Vector2:
        """
        Calculate herding acceleration: a_s = k_c(x_prey - x_s) - k_d*v_s
        
        Args:
            prey_centroid: Center of prey group
            
        Returns:
            Herding acceleration vector
        """
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        centering_force = prey_centroid - self.position
        centering_force = centering_force * self.config['K_C']
        damping_force = -self.velocity * self.config['K_D']
        
        return centering_force + damping_force
    
    def calculate_interception(self, target) -> pygame.Vector2:
        """
        Calculate the interception point where predator can catch target.
        
        Uses predictive targeting based on target's velocity.
        
        Args:
            target: Target boid to intercept
            
        Returns:
            Position to aim for (interception point)
        """
        to_target = target.position - self.position
        distance = to_target.length()
        
        if distance < 0.001:
            return target.position
        
        target_speed = target.velocity.length()
        
        if target_speed < 0.001:
            return target.position
        
        relative_vel = target.velocity - self.velocity
        closing_speed = -to_target.dot(relative_vel) / distance
        
        if closing_speed <= 0:
            return target.position
        
        time_to_intercept = distance / (closing_speed + self.max_speed)
        time_to_intercept = min(time_to_intercept, 180)
        
        return target.position + (target.velocity * time_to_intercept)
    
    def burst_interception(self, boids: list) -> pygame.Vector2:
        """
        Burst with predictive interception toward closest prey.
        
        Args:
            boids: List of boid agents
            
        Returns:
            Acceleration vector toward interception point
        """
        if not boids:
            return pygame.Vector2(0, 0)
        
        closest_prey = min(boids, key=lambda b: self.position.distance_to(b.position))
        
        if self.config.get("predatorInterception", True):
            target_pos = self.calculate_interception(closest_prey)
        else:
            target_pos = closest_prey.position
        
        direction = target_pos - self.position
        if direction.length() > 0.001:
            return direction * self.config['K_B']
        
        return pygame.Vector2(0, 0)
    
    def constrain_turning(self, desired_velocity: pygame.Vector2) -> pygame.Vector2:
        """
        Constrain velocity direction change for realistic movement.
        
        Args:
            desired_velocity: The desired velocity vector
            
        Returns:
            Constrained velocity vector
        """
        current_speed = self.velocity.length()
        desired_speed = desired_velocity.length()
        
        if desired_speed < 0.001:
            return desired_velocity
        
        if current_speed < self.config['minSpeedForTurning']:
            return desired_velocity
        
        if current_speed > 0.001:
            current_direction = math.atan2(self.velocity.y, self.velocity.x)
        else:
            current_direction = self.previous_direction
        
        desired_direction = math.atan2(desired_velocity.y, desired_velocity.x)
        angle_diff = desired_direction - current_direction
        
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Calculate max angular velocity based on speed
        speed_ratio = current_speed / max(self.max_speed, 0.001)
        speed_ratio = max(0.0, min(1.0, speed_ratio))
        max_angular = (
            self.config['maxAngularVelocitySlow'] * (1 - speed_ratio) +
            self.config['maxAngularVelocityFast'] * speed_ratio
        )
        
        if abs(angle_diff) > max_angular:
            angle_diff = math.copysign(max_angular, angle_diff)
        
        constrained_direction = current_direction + angle_diff
        return pygame.Vector2(
            math.cos(constrained_direction) * desired_speed,
            math.sin(constrained_direction) * desired_speed
        )
    
    def update_state(self, boids: list, current_time: int) -> None:
        """
        Update predator state based on prey density and burst duration.
        
        Args:
            boids: List of boid agents
            current_time: Current simulation time in milliseconds
        """
        prey_density = self.calculate_prey_density(boids)
        
        if self.state == "herding":
            if prey_density >= self.config['preyDensityThreshold']:
                self.state = "burst"
                self.max_speed = self.config['burstSpeed']
                self.burst_start_time = current_time
        elif self.state == "burst":
            if self.burst_start_time and (current_time - self.burst_start_time) >= self.config['burstDuration']:
                self.state = "herding"
                self.max_speed = self.config['cruiseSpeed']
                self.burst_start_time = None
    
    def hunt(self, boids: list):
        """
        Individual hunting behavior (non-cooperative).
        
        Args:
            boids: List of boid agents
            
        Returns:
            Caught boid or None
        """
        caught_boid = None
        
        if not boids:
            return caught_boid
        
        current_time = pygame.time.get_ticks()
        self.update_state(boids, current_time)
        
        prey_centroid, _ = self.calculate_prey_centroid(boids)
        
        if self.state == "herding":
            if prey_centroid is not None:
                acceleration = self.herding_acceleration(prey_centroid)
                self.apply_force(acceleration)
        elif self.state == "burst":
            acceleration = self.burst_interception(boids)
            self.apply_force(acceleration)
        
        # Check for catches
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < self.config["catchingDistance"]:
                caught_boid = boid
                self.catch_flash = 15
                break
        
        # Boundary avoidance
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def cooperative_hunt(self, boids: list, other_predators: list, hunting_pack) -> object:
        """
        Cooperative hunting with pack coordination.
        
        Args:
            boids: List of boid agents
            other_predators: List of other predator agents
            hunting_pack: HuntingPack coordinator object
            
        Returns:
            Caught boid or None
        """
        caught_boid = None
        
        if not boids:
            return caught_boid
        
        # Get prey centroid from pack or calculate locally
        if hunting_pack and hunting_pack.prey_centroid:
            prey_centroid = hunting_pack.prey_centroid
        else:
            prey_centroid, _ = self.calculate_prey_centroid(boids)
        
        if self.state == "herding":
            if hunting_pack:
                formation_pos = hunting_pack.get_formation_position(self)
                if formation_pos:
                    formation_force = self.formation_acceleration(formation_pos, prey_centroid)
                    self.apply_force(formation_force)
                    
                    compression_force = self.compression_acceleration(prey_centroid)
                    self.apply_force(compression_force)
            else:
                if prey_centroid is not None:
                    acceleration = self.herding_acceleration(prey_centroid)
                    self.apply_force(acceleration)
            
            separation_force = self.predator_separation(other_predators)
            self.apply_force(separation_force)
        
        elif self.state == "burst":
            acceleration = self.burst_interception(boids)
            self.apply_force(acceleration)
        
        # Check for catches
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < self.config["catchingDistance"]:
                caught_boid = boid
                self.catch_flash = 15
                break
        
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def formation_acceleration(self, formation_pos: pygame.Vector2, prey_centroid: pygame.Vector2) -> pygame.Vector2:
        """
        Calculate acceleration toward formation position.
        
        Args:
            formation_pos: Target position in formation
            prey_centroid: Center of prey group
            
        Returns:
            Formation acceleration vector
        """
        to_formation = formation_pos - self.position
        dist_to_formation = to_formation.length()
        
        speed_boost = 1.0
        if dist_to_formation > 100:
            speed_boost = 1.5
            if self.max_speed < self.config['burstSpeed'] * 0.75:
                self.max_speed = self.config['burstSpeed'] * 0.75
        
        formation_force = to_formation * self.config['K_FORMATION'] * speed_boost
        
        # Add tangential component when close to position
        if dist_to_formation < 80:
            to_prey = prey_centroid - self.position
            if to_prey.length() > 0.001:
                tangent = pygame.Vector2(-to_prey.y, to_prey.x)
                tangent = tangent.normalize() * self.config['K_D']
                formation_force += tangent
        
        return formation_force
    
    def compression_acceleration(self, prey_centroid: pygame.Vector2) -> pygame.Vector2:
        """
        Calculate force pushing toward prey to compress the group.
        
        Args:
            prey_centroid: Center of prey group
            
        Returns:
            Compression acceleration vector
        """
        to_prey = prey_centroid - self.position
        dist = to_prey.length()
        
        if dist > self.config['formationRadius']:
            return to_prey.normalize() * self.config['compressionForce']
        elif dist > self.config['formationRadius'] * 0.5:
            return to_prey.normalize() * self.config['compressionForce'] * 1.5
        
        return pygame.Vector2(0, 0)
    
    def predator_separation(self, other_predators: list) -> pygame.Vector2:
        """
        Maintain separation from other predators.
        
        Args:
            other_predators: List of other predator agents
            
        Returns:
            Separation steering force
        """
        steering = pygame.Vector2(0, 0)
        count = 0
        
        for other in other_predators:
            if other is self:
                continue
            
            dist = self.position.distance_to(other.position)
            if dist < self.config['minPredatorSeparation'] and dist > 0:
                diff = self.position - other.position
                diff = diff / dist
                steering += diff
                count += 1
        
        if count > 0:
            steering = steering / count
            steering = steering * self.config['K_SEPARATION']
        
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
            steering.x = (BOUNDARY_MARGIN - self.position.x) / BOUNDARY_MARGIN
        elif self.position.x > width - BOUNDARY_MARGIN:
            steering.x = -(self.position.x - (width - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
        
        if self.position.y < BOUNDARY_MARGIN:
            steering.y = (BOUNDARY_MARGIN - self.position.y) / BOUNDARY_MARGIN
        elif self.position.y > height - BOUNDARY_MARGIN:
            steering.y = -(self.position.y - (height - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
        
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def update(self) -> None:
        """Update predator with turning constraints."""
        if self.config.get("movementNoise", 0) > 0:
            noise = pygame.Vector2(
                random.uniform(-self.config["movementNoise"], self.config["movementNoise"]),
                random.uniform(-self.config["movementNoise"], self.config["movementNoise"])
            )
            self.apply_force(noise)
        
        desired_velocity = self.velocity + self.acceleration
        
        if desired_velocity.length() > 0:
            constrained_velocity = self.constrain_turning(desired_velocity)
            self.velocity = constrained_velocity
        
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        
        self.position += self.velocity
        
        # Track distance traveled
        self.distance_traveled += self.position.distance_to(self.previous_position)
        self.previous_position = self.position.copy()
        
        if self.velocity.length() > 0:
            self.previous_direction = math.atan2(self.velocity.y, self.velocity.x)
        
        self.acceleration *= 0
        
        if self.catch_flash > 0:
            self.catch_flash -= 1
    
    def draw(self, surface, debug_mode: int = 0, formation_pos=None, boids=None) -> None:
        """
        Draw the predator on the given surface.
        
        Args:
            surface: Pygame surface to draw on
            debug_mode: 0=normal, 1=velocity, 2=full debug
            formation_pos: Target formation position (for debug)
            boids: List of boids (for interception debug)
        """
        if self.catch_flash > 0:
            radius = 6 + int(self.catch_flash * 0.6)
            pygame.draw.circle(surface, (255, 255, 255), 
                             (int(self.position.x), int(self.position.y)), radius)
        else:
            if self.state == "burst":
                draw_color = (255, 100, 100)
            else:
                draw_color = self.config["predatorColor"]
            
            pygame.draw.circle(surface, draw_color, 
                             (int(self.position.x), int(self.position.y)), 6)
            pygame.draw.circle(surface, (200, 0, 0), 
                             (int(self.position.x), int(self.position.y)), 6, 2)
        
        if debug_mode >= 2:
            # Perception radius
            pygame.draw.circle(surface, (100, 50, 50), 
                             (int(self.position.x), int(self.position.y)), 
                             int(self.perception), 1)
            
            # Formation target
            if formation_pos and self.state == "herding":
                pygame.draw.circle(surface, (255, 255, 0), 
                                 (int(formation_pos.x), int(formation_pos.y)), 5, 2)
                pygame.draw.line(surface, (255, 255, 0), self.position, formation_pos, 1)
            
            # Interception point
            if self.state == "burst" and boids and self.config.get("predatorInterception", True):
                closest_prey = min(boids, key=lambda b: self.position.distance_to(b.position))
                intercept_pos = self.calculate_interception(closest_prey)
                pygame.draw.line(surface, (255, 150, 0), self.position, intercept_pos, 2)
                size = 5
                pygame.draw.line(surface, (255, 200, 0), 
                               (intercept_pos.x - size, intercept_pos.y - size),
                               (intercept_pos.x + size, intercept_pos.y + size), 2)
                pygame.draw.line(surface, (255, 200, 0),
                               (intercept_pos.x + size, intercept_pos.y - size),
                               (intercept_pos.x - size, intercept_pos.y + size), 2)

