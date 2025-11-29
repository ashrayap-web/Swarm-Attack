import pygame
import random
import math
import sys
import json
from collections import defaultdict

# ==================== SIMULATION PARAMETERS ====================
CONFIG = {
    "screenWidth": 1200,
    "screenHeight": 650,
    "boidCount": 100,
    "predatorCount": 5,
    "maxSpeed": 4.0,
    "maxForce": 0.1,
    "neighborRadius": 50,
    "desiredSeparation": 20,
    "predatorAvoidRadius": 80,
    "catchingDistance": 8,  # How close predator must be to eat a boid
    "predatorInterception": True,  # Use predictive interception vs simple chase
    "gridCellSize": 80,
    "fpsTarget": 60,
    "speedMultiplier": 1.0,
    "visualizeGrid": False,
    "visualizationMode": 0,  # 0=normal, 1=grid, 2=debug
    "backgroundColor": [25, 25, 25],
    "boidColor": [200, 200, 255],
    "predatorColor": [255, 50, 50],
    "maxAgentCount": 250,
    "accelerationFactor": 1.0,
    "environmentalWind": [0.0, 0.0],
    "adaptiveBehaviorEnabled": True,
    "learningRate": 0.05,
    "splitMergeEnabled": True,
    "scoreOutputFile": "boids_simulation_score.json",
    
    # Shark/Predator herding parameters
    "predatorPerception": 180,
    "K_C": 0.2,
    "K_D": 0.1,
    "K_B": 0.3,
    "preyDensityThreshold": 20,
    "burstDuration": 1000,
    "cruiseSpeed": 2,
    "burstSpeed": 4.0,
    "maxAngularVelocitySlow": 0.08,
    "maxAngularVelocityFast": 0.03,
    "minSpeedForTurning": 0.1,
    
    # Cooperative hunting parameters
    "cooperativeHunting": True,
    "formationRadius": 150,  # Radius for predator positioning around prey
    "formationQualityThreshold": 0.6,  # How well predators must be positioned (0-1)
    "K_FORMATION": 0.15,  # Weight for moving toward assigned formation position
    "K_SEPARATION": 0.05,  # Weight for maintaining distance from other predators
    "minPredatorSeparation": 100,  # Minimum distance between predators
    "compressionForce": 0.1,  # Force pushing toward prey to compress the group
    
    # Smart clustering parameters
    "enableClustering": True,  # Use clustering to find actual fish groups
    "clusterRadius": 60,  # Max distance between fish in same cluster
    "minClusterSize": 5,  # Minimum fish to form a valid target cluster
    "targetStrategy": "largest",  # "nearest", "largest", or "densest"
    "maxTargetDistance": 400,  # Max distance to consider a cluster
}

# Extract commonly used values
WIDTH = CONFIG["screenWidth"]
HEIGHT = CONFIG["screenHeight"]
FPS = CONFIG["fpsTarget"]
GRID_SIZE = CONFIG["gridCellSize"]

# Rule weights
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.3
COHESION_WEIGHT = 1.0
BOUNDARY_WEIGHT = 4
PREDATOR_AVOID_WEIGHT = 3.0

BOUNDARY_MARGIN = 80

# ==================== SPATIAL GRID ====================
class SpatialGrid:
    """Spatial hash grid for efficient neighbor lookup"""
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
    
    def clear(self):
        self.grid.clear()
    
    def _hash(self, x, y):
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (max(0, min(col, self.cols - 1)), max(0, min(row, self.rows - 1)))
    
    def insert(self, agent):
        cell = self._hash(agent.position.x, agent.position.y)
        self.grid[cell].append(agent)
    
    def get_neighbors(self, position, radius):
        """Get all agents within radius of position"""
        neighbors = []
        cell = self._hash(position.x, position.y)
        cells_to_check = self._get_adjacent_cells(cell)
        
        for c in cells_to_check:
            for agent in self.grid.get(c, []):
                dist = position.distance_to(agent.position)
                if dist < radius:
                    neighbors.append(agent)
        
        return neighbors
    
    def _get_adjacent_cells(self, cell):
        """Get cell and its 8 neighbors"""
        col, row = cell
        cells = []
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                nc, nr = col + dc, row + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    cells.append((nc, nr))
        return cells

# ==================== BASE AGENT CLASS ====================
class Agent:
    def __init__(self, x, y, max_speed, max_force):
        self.position = pygame.Vector2(x, y)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1.0, max_speed)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = max_speed * CONFIG["speedMultiplier"]
        self.max_force = max_force * CONFIG["accelerationFactor"]
    
    def apply_force(self, force):
        self.acceleration += force
    
    def update(self):
        # Apply environmental wind
        wind = pygame.Vector2(CONFIG["environmentalWind"])
        self.apply_force(wind)
        
        self.velocity += self.acceleration
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        
        self.position += self.velocity
        self.acceleration *= 0
    
    def steering(self, desired):
        steer = desired - self.velocity
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

# ==================== BOID CLASS ====================
class Boid(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, CONFIG["maxSpeed"], CONFIG["maxForce"])
        self.color = CONFIG["boidColor"]
        self.adaptive_weights = {
            "separation": SEPARATION_WEIGHT,
            "alignment": ALIGNMENT_WEIGHT,
            "cohesion": COHESION_WEIGHT
        }
        self.stress_level = 0.0  # for adaptive behavior
    
    def flock(self, spatial_grid, predators):
        neighbors = spatial_grid.get_neighbors(self.position, CONFIG["neighborRadius"])
        neighbors = [n for n in neighbors if isinstance(n, Boid) and n is not self]
        
        # Basic boid rules
        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        
        # Boundary repulsion
        boundary = self.boundary_repulsion()
        
        # Predator avoidance
        pred_avoid = self.avoid_predators(predators)
        
        # Apply adaptive weights if enabled
        if CONFIG["adaptiveBehaviorEnabled"]:
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
    
    def separation(self, neighbors):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in neighbors:
            dist = self.position.distance_to(other.position)
            if 0 < dist < CONFIG["desiredSeparation"]:
                diff = self.position - other.position
                diff /= dist  # weight by distance
                steering += diff
                total += 1
        
        if total > 0:
            steering /= total
            if steering.length() > 0.001:
                steering.scale_to_length(self.max_speed)
                steering = self.steering(steering)
        return steering
    
    def alignment(self, neighbors):
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
    
    def cohesion(self, neighbors):
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
    
    def boundary_repulsion(self):
        steering = pygame.Vector2(0, 0)
        
        if self.position.x < BOUNDARY_MARGIN:
            force = pygame.Vector2(1, 0)
            force *= (BOUNDARY_MARGIN - self.position.x) / BOUNDARY_MARGIN
            steering += force
        
        if self.position.x > WIDTH - BOUNDARY_MARGIN:
            force = pygame.Vector2(-1, 0)
            force *= (self.position.x - (WIDTH - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force
        
        if self.position.y < BOUNDARY_MARGIN:
            force = pygame.Vector2(0, 1)
            force *= (BOUNDARY_MARGIN - self.position.y) / BOUNDARY_MARGIN
            steering += force
        
        if self.position.y > HEIGHT - BOUNDARY_MARGIN:
            force = pygame.Vector2(0, -1)
            force *= (self.position.y - (HEIGHT - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force
        
        # Only apply steering if we have a significant force
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def avoid_predators(self, predators):
        steering = pygame.Vector2(0, 0)
        total = 0
        
        for pred in predators:
            dist = self.position.distance_to(pred.position)
            if dist < CONFIG["predatorAvoidRadius"]:
                diff = self.position - pred.position
                diff /= (dist + 1)  # weight by distance
                steering += diff
                total += 1
                self.stress_level = min(1.0, self.stress_level + 0.1)
        
        if total > 0:
            steering /= total
            if steering.length() > 0.001:
                steering.scale_to_length(self.max_speed * 1.5)  # flee faster
                steering = self.steering(steering)
        else:
            self.stress_level = max(0.0, self.stress_level - 0.02)
        
        return steering
    
    def adapt_behavior(self, neighbors, predators):
        """Adaptive behavior based on environmental conditions"""
        if not CONFIG["adaptiveBehaviorEnabled"]:
            return
        
        lr = CONFIG["learningRate"]
        
        # If under stress (predators nearby), increase separation
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
    
    def draw(self, surface, debug_mode=False):
        # Draw as circle
        color = self.color
        if debug_mode == 2:  # debug mode shows stress
            stress_color = int(255 * self.stress_level)
            color = (stress_color, self.color[1], self.color[2])
        
        pygame.draw.circle(surface, color, (int(self.position.x), int(self.position.y)), 4)
        
        # Draw velocity direction
        if debug_mode >= 1:
            end_pos = self.position + self.velocity.normalize() * 10
            pygame.draw.line(surface, color, self.position, end_pos, 1)

# ==================== PREY CLUSTERING ====================
class PreyCluster:
    """Represents a cluster/group of prey fish"""
    def __init__(self, fish_list):
        self.fish = fish_list
        self.centroid = self._calculate_centroid()
        self.size = len(fish_list)
        self.radius = self._calculate_radius()
        self.density = self._calculate_density()
    
    def _calculate_centroid(self):
        """Calculate the center of mass of the cluster"""
        if not self.fish:
            return pygame.Vector2(0, 0)
        center = pygame.Vector2(0, 0)
        for fish in self.fish:
            center += fish.position
        return center / len(self.fish)
    
    def _calculate_radius(self):
        """Calculate the spread/radius of the cluster"""
        if not self.fish:
            return 0
        max_dist = 0
        for fish in self.fish:
            dist = self.centroid.distance_to(fish.position)
            if dist > max_dist:
                max_dist = dist
        return max_dist
    
    def _calculate_density(self):
        """Calculate density (fish per unit area)"""
        if self.radius == 0:
            return float('inf') if self.size > 0 else 0
        area = math.pi * self.radius * self.radius
        return self.size / area if area > 0 else 0
    
    def distance_to(self, position):
        """Distance from a position to this cluster's centroid"""
        return self.centroid.distance_to(position)
    
    def update(self):
        """Recalculate cluster properties"""
        self.centroid = self._calculate_centroid()
        self.size = len(self.fish)
        self.radius = self._calculate_radius()
        self.density = self._calculate_density()

def detect_prey_clusters(boids):
    """
    Detect clusters of prey using distance-based clustering (similar to DBSCAN).
    Returns a list of PreyCluster objects sorted by size.
    """
    if not boids or not CONFIG["enableClustering"]:
        # Fallback: treat all boids as one cluster
        return [PreyCluster(boids)] if boids else []
    
    cluster_radius = CONFIG["clusterRadius"]
    min_cluster_size = CONFIG["minClusterSize"]
    
    # Track which boids have been assigned to clusters
    unassigned = set(boids)
    clusters = []
    
    while unassigned:
        # Start a new cluster with an arbitrary boid
        seed = unassigned.pop()
        current_cluster = [seed]
        
        # Expand cluster by finding neighbors within cluster_radius
        # Use a queue-based approach for efficiency
        to_check = [seed]
        
        while to_check:
            checking_boid = to_check.pop(0)
            
            # Find all unassigned boids within cluster_radius
            neighbors = []
            for boid in list(unassigned):  # Copy to avoid modification during iteration
                dist = checking_boid.position.distance_to(boid.position)
                if dist < cluster_radius:
                    neighbors.append(boid)
            
            # Add neighbors to cluster and queue for checking
            for neighbor in neighbors:
                if neighbor in unassigned:
                    current_cluster.append(neighbor)
                    unassigned.remove(neighbor)
                    to_check.append(neighbor)
        
        # Create cluster object (even for small groups, but mark them)
        if len(current_cluster) >= min_cluster_size:
            clusters.append(PreyCluster(current_cluster))
    
    # Sort by size (largest first)
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters

# ==================== HUNTING PACK CLASS (COORDINATED HUNTING) ====================
class HuntingPack:
    """Manages coordinated hunting behavior among multiple predators"""
    def __init__(self, predators):
        self.predators = predators
        self.collective_state = "herding"  # "herding" or "burst"
        self.burst_start_time = None
        self.prey_centroid = None
        self.formation_quality = 0.0
        self.assigned_angles = {}  # Maps predator to target angle around prey
        self.target_cluster = None  # The specific cluster being targeted
        self.all_clusters = []  # All detected clusters
    
    def update_collective_state(self, boids, current_time):
        """Update the collective hunting state for all predators"""
        if not boids or not self.predators:
            return
        
        # Detect prey clusters using smart clustering
        if CONFIG["enableClustering"]:
            self.all_clusters = detect_prey_clusters(boids)
            
            # Select target cluster based on strategy
            self.target_cluster = self._select_target_cluster()
            
            # Use target cluster centroid, or fallback to global centroid
            if self.target_cluster:
                self.prey_centroid = self.target_cluster.centroid
            else:
                # Fallback to global centroid if no valid cluster
                self.prey_centroid = pygame.Vector2(0, 0)
                for boid in boids:
                    self.prey_centroid += boid.position
                self.prey_centroid /= len(boids)
        else:
            # Use global centroid (old behavior)
            self.prey_centroid = pygame.Vector2(0, 0)
            for boid in boids:
                self.prey_centroid += boid.position
            self.prey_centroid /= len(boids)
            self.target_cluster = PreyCluster(boids)
            self.all_clusters = [self.target_cluster]
        
        # Assign formation positions to each predator
        self._assign_formation_positions()
        
        # Calculate how well predators are positioned
        self.formation_quality = self._calculate_formation_quality()
        
        # Calculate prey density near target centroid
        if self.target_cluster:
            prey_density = self.target_cluster.size
        else:
            prey_density = self._calculate_global_prey_density(boids)
        
        # State transitions
        if self.collective_state == "herding":
            # Burst when: good formation AND high prey density
            if (self.formation_quality >= CONFIG['formationQualityThreshold'] and 
                prey_density >= CONFIG['preyDensityThreshold']):
                self.collective_state = "burst"
                self.burst_start_time = current_time
                # Update all predators to burst mode
                for pred in self.predators:
                    pred.state = "burst"
                    pred.max_speed = CONFIG['burstSpeed']
        
        elif self.collective_state == "burst":
            # Return to herding after burst duration
            if self.burst_start_time and (current_time - self.burst_start_time) >= CONFIG['burstDuration']:
                self.collective_state = "herding"
                self.burst_start_time = None
                # Update all predators back to herding
                for pred in self.predators:
                    pred.state = "herding"
                    pred.max_speed = CONFIG['cruiseSpeed']
    
    def _select_target_cluster(self):
        """Select which cluster to target based on strategy"""
        if not self.all_clusters:
            return None
        
        # Filter clusters that are within max distance of any predator
        valid_clusters = []
        max_dist = CONFIG["maxTargetDistance"]
        
        for cluster in self.all_clusters:
            # Check if at least one predator is close enough
            for pred in self.predators:
                if cluster.distance_to(pred.position) < max_dist:
                    valid_clusters.append(cluster)
                    break
        
        if not valid_clusters:
            # If no clusters in range, target the nearest one
            valid_clusters = self.all_clusters
        
        strategy = CONFIG["targetStrategy"]
        
        if strategy == "largest":
            # Target the largest cluster
            return max(valid_clusters, key=lambda c: c.size)
        
        elif strategy == "densest":
            # Target the densest cluster
            return max(valid_clusters, key=lambda c: c.density)
        
        else:  # "nearest" or default
            # Target the nearest cluster to pack centroid
            pack_center = pygame.Vector2(0, 0)
            for pred in self.predators:
                pack_center += pred.position
            pack_center /= len(self.predators)
            
            return min(valid_clusters, key=lambda c: c.distance_to(pack_center))
    
    def _assign_formation_positions(self):
        """Assign each predator a target angle around the prey centroid"""
        if not self.predators:
            return
        
        num_predators = len(self.predators)
        angle_spacing = (2 * math.pi) / num_predators
        
        # Sort predators by their current angle to maintain smooth transitions
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
    
    def _calculate_formation_quality(self):
        """Calculate how well predators are distributed around prey (0-1)"""
        if not self.predators or len(self.predators) < 2:
            return 1.0  # Perfect if only one predator
        
        # Check if predators are roughly evenly distributed around prey
        ideal_spacing = (2 * math.pi) / len(self.predators)
        
        # Calculate actual angles of predators relative to prey
        actual_angles = []
        for pred in self.predators:
            to_prey = self.prey_centroid - pred.position
            if to_prey.length() > 0.001:
                angle = math.atan2(to_prey.y, to_prey.x)
                actual_angles.append(angle)
        
        if not actual_angles:
            return 0.0
        
        actual_angles.sort()
        
        # Calculate spacing variance
        spacing_errors = []
        for i in range(len(actual_angles)):
            next_i = (i + 1) % len(actual_angles)
            spacing = actual_angles[next_i] - actual_angles[i]
            if spacing < 0:
                spacing += 2 * math.pi
            error = abs(spacing - ideal_spacing)
            spacing_errors.append(error)
        
        # Average error (lower is better)
        avg_error = sum(spacing_errors) / len(spacing_errors) if spacing_errors else 0
        
        # Convert to quality score (0-1, where 1 is perfect)
        quality = 1.0 - min(1.0, avg_error / math.pi)
        
        # Also check if predators are at appropriate distance from prey
        distance_quality = 0.0
        for pred in self.predators:
            dist = pred.position.distance_to(self.prey_centroid)
            ideal_dist = CONFIG['formationRadius']
            dist_error = abs(dist - ideal_dist) / ideal_dist
            distance_quality += max(0, 1.0 - dist_error)
        distance_quality /= len(self.predators)
        
        # Combine spacing and distance quality
        return (quality * 0.6 + distance_quality * 0.4)
    
    def _calculate_global_prey_density(self, boids):
        """Calculate density of prey around the global centroid"""
        if not self.prey_centroid:
            return 0
        
        count = 0
        radius = CONFIG['predatorPerception']
        for boid in boids:
            dist = self.prey_centroid.distance_to(boid.position)
            if dist < radius:
                count += 1
        
        return count
    
    def get_formation_position(self, predator):
        """Get the target formation position for a specific predator"""
        if predator not in self.assigned_angles or not self.prey_centroid:
            return None
        
        target_angle = self.assigned_angles[predator]
        formation_radius = CONFIG['formationRadius']
        
        # Calculate position at the assigned angle around prey centroid
        target_x = self.prey_centroid.x + formation_radius * math.cos(target_angle)
        target_y = self.prey_centroid.y + formation_radius * math.sin(target_angle)
        
        return pygame.Vector2(target_x, target_y)

# ==================== PREDATOR CLASS (HERDING BEHAVIOR) ====================
class Predator(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, CONFIG["cruiseSpeed"], CONFIG["maxForce"] * 1.2)
        self.color = CONFIG["predatorColor"]
        self.catch_flash = 0  # Visual feedback when catching prey
        
        # Shark behavior state
        self.state = "herding"  # "herding" or "burst"
        self.burst_start_time = None  # Track when burst mode started
        self.perception = CONFIG["predatorPerception"]
        
        # Store previous velocity direction for realistic turning
        if self.velocity.length() > 0:
            self.previous_direction = math.atan2(self.velocity.y, self.velocity.x)
        else:
            self.previous_direction = 0.0  # Default initial direction
    
    def calculate_prey_centroid(self, boids):
        """Calculate the centroid of prey within perception radius."""
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
    
    def calculate_prey_density(self, boids):
        """Calculate the number of prey within perception radius."""
        count = 0
        perception_sq = self.perception * self.perception
        
        for boid in boids:
            dist_sq = self.position.distance_squared_to(boid.position)
            if dist_sq < perception_sq:
                count += 1
        
        return count
    
    def herding_acceleration(self, prey_centroid):
        """Herding acceleration: a_s = k_c(x_prey_centroid - x_s) - k_d*v_s"""
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        # Centering force toward prey centroid
        centering_force = prey_centroid - self.position
        centering_force = centering_force * CONFIG['K_C']
        
        # Damping force (opposes velocity)
        damping_force = -self.velocity * CONFIG['K_D']
        
        # Total acceleration
        acceleration = centering_force + damping_force
        return acceleration
    
    def calculate_interception(self, target):
        """
        Calculate the interception point where predator can catch target.
        Uses predictive targeting based on target's velocity.
        
        Returns the position to aim for (interception point).
        """
        # Get relative position and velocities
        to_target = target.position - self.position
        distance = to_target.length()
        
        if distance < 0.001:
            return target.position
        
        # Calculate time to intercept
        target_speed = target.velocity.length()
        
        # If target is stationary, aim directly at it
        if target_speed < 0.001:
            return target.position
        
        # Calculate the interception time using closing speed approach
        # Dot product to see if target is moving toward or away from us
        relative_vel = target.velocity - self.velocity
        closing_speed = -to_target.dot(relative_vel) / distance
        
        # If moving apart or very slow closing, use simple chase
        if closing_speed <= 0:
            return target.position
        
        # Estimate interception time
        time_to_intercept = distance / (closing_speed + self.max_speed)
        
        # Clamp time to reasonable values (max 3 seconds of prediction)
        time_to_intercept = min(time_to_intercept, 180)  # 3 seconds at 60 FPS
        
        # Calculate where target will be
        predicted_position = target.position + (target.velocity * time_to_intercept)
        
        return predicted_position
    
    def burst_acceleration(self, prey_centroid):
        """Burst acceleration: a_s = k_b(x_prey_centroid - x_s)"""
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        # Direct acceleration toward prey centroid
        direction = prey_centroid - self.position
        acceleration = direction * CONFIG['K_B']
        return acceleration
    
    def burst_interception(self, boids):
        """
        Burst with predictive interception: target closest prey and predict its position.
        Returns acceleration vector toward interception point.
        """
        if not boids:
            return pygame.Vector2(0, 0)
        
        # Find closest prey
        closest_prey = min(boids, key=lambda b: self.position.distance_to(b.position))
        
        # Use interception if enabled, otherwise direct pursuit
        if CONFIG["predatorInterception"]:
            target_pos = self.calculate_interception(closest_prey)
        else:
            target_pos = closest_prey.position
        
        # Calculate acceleration toward interception point
        direction = target_pos - self.position
        if direction.length() > 0.001:
            acceleration = direction * CONFIG['K_B']
            return acceleration
        
        return pygame.Vector2(0, 0)
    
    def constrain_turning(self, desired_velocity):
        """Constrain velocity direction change based on current speed for realistic movement."""
        current_speed = self.velocity.length()
        desired_speed = desired_velocity.length()
        
        # If desired velocity is zero, return as is
        if desired_speed < 0.001:
            return desired_velocity
        
        # If moving very slowly, allow free turning
        if current_speed < CONFIG['minSpeedForTurning']:
            return desired_velocity
        
        # Calculate current and desired directions
        if current_speed > 0.001:
            current_direction = math.atan2(self.velocity.y, self.velocity.x)
        else:
            current_direction = self.previous_direction
        
        desired_direction = math.atan2(desired_velocity.y, desired_velocity.x)
        
        # Calculate angular difference (handling wrap-around)
        angle_diff = desired_direction - current_direction
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Calculate maximum angular velocity based on current speed
        # Interpolate between slow and fast turning rates based on speed
        speed_ratio = current_speed / max(self.max_speed, 0.001)  # Avoid division by zero
        speed_ratio = max(0.0, min(1.0, speed_ratio))  # Clamp to [0, 1]
        max_angular_velocity = CONFIG['maxAngularVelocitySlow'] * (1 - speed_ratio) + CONFIG['maxAngularVelocityFast'] * speed_ratio
        
        # Limit the angular change
        if abs(angle_diff) > max_angular_velocity:
            angle_diff = math.copysign(max_angular_velocity, angle_diff)
        
        # Apply the constrained angle change, preserving desired speed
        constrained_direction = current_direction + angle_diff
        constrained_velocity = pygame.Vector2(
            math.cos(constrained_direction) * desired_speed,
            math.sin(constrained_direction) * desired_speed
        )
        
        return constrained_velocity
    
    def update_state(self, boids, current_time):
        """Update shark state based on prey density and burst duration."""
        prey_density = self.calculate_prey_density(boids)
        
        if self.state == "herding":
            # Transition to burst when prey density exceeds threshold
            if prey_density >= CONFIG['preyDensityThreshold']:
                self.state = "burst"
                self.max_speed = CONFIG['burstSpeed']
                self.burst_start_time = current_time
        elif self.state == "burst":
            # Transition back to herding after burst duration
            if self.burst_start_time and (current_time - self.burst_start_time) >= CONFIG['burstDuration']:
                self.state = "herding"
                self.max_speed = CONFIG['cruiseSpeed']
                self.burst_start_time = None
    
    def cooperative_hunt(self, boids, other_predators, hunting_pack):
        """Enhanced hunting with cooperative behavior"""
        caught_boid = None
        
        if not boids:
            return caught_boid
        
        # Calculate prey centroid (local or from pack)
        if hunting_pack and hunting_pack.prey_centroid:
            prey_centroid = hunting_pack.prey_centroid
        else:
            prey_centroid, _ = self.calculate_prey_centroid(boids)
        
        # Apply appropriate acceleration based on state
        if self.state == "herding":
            # Cooperative herding: move to formation position
            if hunting_pack:
                formation_pos = hunting_pack.get_formation_position(self)
                if formation_pos:
                    # Move toward formation position
                    formation_force = self.formation_acceleration(formation_pos, prey_centroid)
                    self.apply_force(formation_force)
                    
                    # Add compression force to push prey together
                    compression_force = self.compression_acceleration(prey_centroid)
                    self.apply_force(compression_force)
            else:
                # Fallback to basic herding
                if prey_centroid is not None:
                    acceleration = self.herding_acceleration(prey_centroid)
                    self.apply_force(acceleration)
            
            # Maintain separation from other predators
            separation_force = self.predator_separation(other_predators)
            self.apply_force(separation_force)
        
        elif self.state == "burst":
            # Burst mode: intercept closest prey
            acceleration = self.burst_interception(boids)
            self.apply_force(acceleration)
        
        # Check for collisions (shark eating fish)
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < CONFIG["catchingDistance"]:
                caught_boid = boid
                self.catch_flash = 15  # Flash for 15 frames
                break
        
        # Boundary avoidance
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def hunt(self, boids):
        """Shark behavior: herding or burst based on state (non-cooperative)."""
        caught_boid = None
        
        if not boids:
            return caught_boid
        
        current_time = pygame.time.get_ticks()
        
        # Update state based on prey density
        self.update_state(boids, current_time)
        
        # Calculate prey centroid
        prey_centroid, prey_count = self.calculate_prey_centroid(boids)
        
        # Apply appropriate acceleration based on state
        if self.state == "herding":
            # Herding mode: a_s = k_c(x_prey_centroid - x_s) - k_d*v_s
            if prey_centroid is not None:
                acceleration = self.herding_acceleration(prey_centroid)
                self.apply_force(acceleration)
        elif self.state == "burst":
            # Burst mode: intercept closest prey
            acceleration = self.burst_interception(boids)
            self.apply_force(acceleration)
        
        # Check for collisions (shark eating fish)
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < CONFIG["catchingDistance"]:
                caught_boid = boid
                self.catch_flash = 15  # Flash for 15 frames
                break
        
        # Boundary avoidance
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def formation_acceleration(self, formation_pos, prey_centroid):
        """Calculate acceleration toward formation position while circling prey"""
        # Move toward formation position
        to_formation = formation_pos - self.position
        formation_force = to_formation * CONFIG['K_FORMATION']
        
        # Add tangential component to circle around prey
        to_prey = prey_centroid - self.position
        if to_prey.length() > 0.001:
            # Calculate tangent (perpendicular to radius)
            tangent = pygame.Vector2(-to_prey.y, to_prey.x)
            tangent = tangent.normalize() * CONFIG['K_D']
            formation_force += tangent
        
        return formation_force
    
    def compression_acceleration(self, prey_centroid):
        """Gentle force pushing toward prey to compress the group"""
        to_prey = prey_centroid - self.position
        dist = to_prey.length()
        
        # Only apply if we're outside the formation radius
        if dist > CONFIG['formationRadius']:
            compression = to_prey.normalize() * CONFIG['compressionForce']
            return compression
        
        return pygame.Vector2(0, 0)
    
    def predator_separation(self, other_predators):
        """Maintain separation from other predators to avoid clustering"""
        steering = pygame.Vector2(0, 0)
        count = 0
        
        for other in other_predators:
            if other is self:
                continue
            
            dist = self.position.distance_to(other.position)
            if dist < CONFIG['minPredatorSeparation'] and dist > 0:
                # Repel from other predator
                diff = self.position - other.position
                diff = diff / dist  # Weight by distance
                steering += diff
                count += 1
        
        if count > 0:
            steering = steering / count
            steering = steering * CONFIG['K_SEPARATION']
        
        return steering
    
    def boundary_repulsion(self):
        steering = pygame.Vector2(0, 0)
        margin = BOUNDARY_MARGIN
        
        if self.position.x < margin:
            steering.x = (margin - self.position.x) / margin
        elif self.position.x > WIDTH - margin:
            steering.x = -(self.position.x - (WIDTH - margin)) / margin
        
        if self.position.y < margin:
            steering.y = (margin - self.position.y) / margin
        elif self.position.y > HEIGHT - margin:
            steering.y = -(self.position.y - (HEIGHT - margin)) / margin
        
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def update(self):
        """Override update to apply realistic turning constraints."""
        # Apply environmental wind
        wind = pygame.Vector2(CONFIG["environmentalWind"])
        self.apply_force(wind)
        
        # Calculate desired velocity (normal update)
        desired_velocity = self.velocity + self.acceleration
        
        # Apply turning constraints before updating velocity
        if desired_velocity.length() > 0:
            constrained_velocity = self.constrain_turning(desired_velocity)
            self.velocity = constrained_velocity
        
        # Limit speed
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        
        # Update position
        self.position += self.velocity
        
        # Store current direction for next frame
        if self.velocity.length() > 0:
            self.previous_direction = math.atan2(self.velocity.y, self.velocity.x)
        
        # Reset acceleration
        self.acceleration *= 0
        
        # Handle catch flash
        if self.catch_flash > 0:
            self.catch_flash -= 1
    
    def draw(self, surface, debug_mode=False, formation_pos=None, boids=None):
        # Draw larger and brighter when catching prey
        if self.catch_flash > 0:
            radius = 6 + int(self.catch_flash * 0.6)  # Grow up to 15 pixels
            flash_color = (255, 255, 255)  # White flash
            pygame.draw.circle(surface, flash_color, (int(self.position.x), int(self.position.y)), radius)
        else:
            # Change color based on state: brighter red during burst
            if self.state == "burst":
                draw_color = (255, 100, 100)  # Brighter red during burst
            else:
                draw_color = self.color  # Normal red during herding
            
            pygame.draw.circle(surface, draw_color, (int(self.position.x), int(self.position.y)), 6)
            pygame.draw.circle(surface, (200, 0, 0), (int(self.position.x), int(self.position.y)), 6, 2)
        
        # Debug mode: show perception radius and formation target
        if debug_mode >= 2:
            pygame.draw.circle(surface, (100, 50, 50), (int(self.position.x), int(self.position.y)), 
                             int(self.perception), 1)
            
            # Show formation target position during herding
            if formation_pos and self.state == "herding":
                pygame.draw.circle(surface, (255, 255, 0), (int(formation_pos.x), int(formation_pos.y)), 5, 2)
                pygame.draw.line(surface, (255, 255, 0), self.position, formation_pos, 1)
            
            # Show interception point during burst
            if self.state == "burst" and boids and CONFIG["predatorInterception"]:
                closest_prey = min(boids, key=lambda b: self.position.distance_to(b.position))
                intercept_pos = self.calculate_interception(closest_prey)
                # Draw line from predator to predicted interception point
                pygame.draw.line(surface, (255, 150, 0), self.position, intercept_pos, 2)
                # Draw X at interception point
                size = 5
                pygame.draw.line(surface, (255, 200, 0), 
                               (intercept_pos.x - size, intercept_pos.y - size),
                               (intercept_pos.x + size, intercept_pos.y + size), 2)
                pygame.draw.line(surface, (255, 200, 0),
                               (intercept_pos.x + size, intercept_pos.y - size),
                               (intercept_pos.x - size, intercept_pos.y + size), 2)

# ==================== SIMULATION CLASS ====================
class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Advanced Boids Simulation")
        self.clock = pygame.time.Clock()
        
        self.spatial_grid = SpatialGrid(WIDTH, HEIGHT, GRID_SIZE)
        
        # Initialize agents
        self.boids = []
        self.predators = []
        
        self._spawn_agents()
        
        # Initialize hunting pack for cooperative behavior
        if CONFIG["cooperativeHunting"] and len(self.predators) > 1:
            self.hunting_pack = HuntingPack(self.predators)
        else:
            self.hunting_pack = None
        
        self.frame_count = 0
        self.running = True
        
        # Statistics
        self.stats = {
            "avg_speed": 0,
            "avg_neighbors": 0,
            "avg_stress": 0,
            "flock_cohesion": 0,
            "boids_eaten": 0,
            "formation_quality": 0,
            "pack_state": "none",
            "num_clusters": 0,
            "target_cluster_size": 0
        }
    
    def _spawn_agents(self):
        # Spawn boids
        for _ in range(CONFIG["boidCount"]):
            x = random.uniform(100, WIDTH - 100)
            y = random.uniform(100, HEIGHT - 100)
            self.boids.append(Boid(x, y))
        
        # Spawn predators
        for _ in range(CONFIG["predatorCount"]):
            x = random.uniform(50, WIDTH - 50)
            y = random.uniform(50, HEIGHT - 50)
            self.predators.append(Predator(x, y))
    
    def update(self):
        # Rebuild spatial grid
        self.spatial_grid.clear()
        for boid in self.boids:
            self.spatial_grid.insert(boid)
        
        # Update boids
        for boid in self.boids:
            boid.flock(self.spatial_grid, self.predators)
            boid.update()
        
        # Update hunting pack coordination
        if self.hunting_pack and CONFIG["cooperativeHunting"]:
            current_time = pygame.time.get_ticks()
            self.hunting_pack.update_collective_state(self.boids, current_time)
        
        # Update predators and check for caught boids
        caught_boids = []
        for predator in self.predators:
            # Use cooperative or individual hunting
            if self.hunting_pack and CONFIG["cooperativeHunting"]:
                caught = predator.cooperative_hunt(self.boids, self.predators, self.hunting_pack)
            else:
                caught = predator.hunt(self.boids)
            
            if caught:
                caught_boids.append(caught)
            predator.update()
        
        # Remove caught boids
        for caught in caught_boids:
            if caught in self.boids:
                self.boids.remove(caught)
                self.stats["boids_eaten"] += 1
        
        # Handle split/merge if enabled
        if CONFIG["splitMergeEnabled"] and self.frame_count % 100 == 0:
            self._handle_split_merge()
        
        self.frame_count += 1
        self._update_statistics()
    
    def _handle_split_merge(self):
        """Split large groups or merge small groups"""
        if len(self.boids) < CONFIG["maxAgentCount"] and random.random() < 0.1:
            # Small chance to add a boid
            x = random.uniform(100, WIDTH - 100)
            y = random.uniform(100, HEIGHT - 100)
            self.boids.append(Boid(x, y))
        elif len(self.boids) > CONFIG["boidCount"] * 1.5:
            # Remove a random boid if too many
            if self.boids:
                self.boids.pop(random.randint(0, len(self.boids) - 1))
    
    def _update_statistics(self):
        if not self.boids:
            return
        
        total_speed = sum(b.velocity.length() for b in self.boids)
        self.stats["avg_speed"] = total_speed / len(self.boids)
        
        total_stress = sum(b.stress_level for b in self.boids)
        self.stats["avg_stress"] = total_stress / len(self.boids)
        
        # Calculate flock cohesion (average distance to centroid)
        if self.boids:
            centroid = pygame.Vector2(0, 0)
            for b in self.boids:
                centroid += b.position
            centroid /= len(self.boids)
            
            total_dist = sum(b.position.distance_to(centroid) for b in self.boids)
            self.stats["flock_cohesion"] = total_dist / len(self.boids)
        
        # Update pack statistics
        if self.hunting_pack:
            self.stats["formation_quality"] = self.hunting_pack.formation_quality
            self.stats["pack_state"] = self.hunting_pack.collective_state
            self.stats["num_clusters"] = len(self.hunting_pack.all_clusters)
            if self.hunting_pack.target_cluster:
                self.stats["target_cluster_size"] = self.hunting_pack.target_cluster.size
            else:
                self.stats["target_cluster_size"] = 0
        else:
            self.stats["formation_quality"] = 0
            self.stats["pack_state"] = "none"
            self.stats["num_clusters"] = 0
            self.stats["target_cluster_size"] = 0
    
    def draw(self):
        self.screen.fill(CONFIG["backgroundColor"])
        
        # Draw grid if enabled
        if CONFIG["visualizeGrid"] or CONFIG["visualizationMode"] == 1:
            self._draw_grid()
        
        # Draw cooperative hunting visualization
        debug_mode = CONFIG["visualizationMode"]
        if debug_mode >= 2 and self.hunting_pack:
            self._draw_hunting_visualization()
        
        # Draw boids
        for boid in self.boids:
            boid.draw(self.screen, debug_mode)
        
        # Draw predators with formation positions
        for predator in self.predators:
            formation_pos = None
            if self.hunting_pack and debug_mode >= 2:
                formation_pos = self.hunting_pack.get_formation_position(predator)
            predator.draw(self.screen, debug_mode, formation_pos, self.boids)
        
        # Draw stats
        self._draw_stats()
        
        pygame.display.flip()
    
    def _draw_hunting_visualization(self):
        """Draw visualization of cooperative hunting behavior"""
        if not self.hunting_pack:
            return
        
        # Draw all detected clusters (light circles)
        if CONFIG["enableClustering"] and self.hunting_pack.all_clusters:
            for cluster in self.hunting_pack.all_clusters:
                # Draw cluster boundary
                color = (80, 80, 80)  # Gray for non-target clusters
                if cluster == self.hunting_pack.target_cluster:
                    color = (100, 200, 100)  # Green for target cluster
                
                # Draw cluster centroid
                pygame.draw.circle(self.screen, color, 
                                 (int(cluster.centroid.x), int(cluster.centroid.y)), 5, 2)
                
                # Draw cluster radius
                if cluster.radius > 0:
                    pygame.draw.circle(self.screen, color, 
                                     (int(cluster.centroid.x), int(cluster.centroid.y)), 
                                     int(cluster.radius), 1)
                
                # Draw cluster size label
                font = pygame.font.Font(None, 18)
                text = font.render(str(cluster.size), True, color)
                self.screen.blit(text, (int(cluster.centroid.x) + 10, int(cluster.centroid.y) - 10))
        
        # Draw target prey centroid (emphasized)
        if self.hunting_pack.prey_centroid:
            centroid = self.hunting_pack.prey_centroid
            pygame.draw.circle(self.screen, (100, 255, 100), 
                             (int(centroid.x), int(centroid.y)), 10, 3)
            
            # Draw formation circle around target
            pygame.draw.circle(self.screen, (150, 200, 50), 
                             (int(centroid.x), int(centroid.y)), 
                             int(CONFIG['formationRadius']), 2)
    
    def _draw_grid(self):
        """Draw spatial grid"""
        for i in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (i, 0), (i, HEIGHT), 1)
        for j in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (0, j), (WIDTH, j), 1)
    
    def _draw_stats(self):
        """Draw statistics on screen"""
        font = pygame.font.Font(None, 24)
        y_offset = 10
        
        stats_text = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Boids: {len(self.boids)}",
            f"Eaten: {self.stats['boids_eaten']}",
            f"Avg Speed: {self.stats['avg_speed']:.2f}",
            f"Avg Stress: {self.stats['avg_stress']:.2f}",
            f"Cohesion: {self.stats['flock_cohesion']:.1f}"
        ]
        
        # Add cooperative hunting stats if enabled
        if CONFIG["cooperativeHunting"] and self.hunting_pack:
            stats_text.append(f"Formation: {self.stats['formation_quality']:.2f}")
            stats_text.append(f"Pack: {self.stats['pack_state']}")
            if CONFIG["enableClustering"]:
                stats_text.append(f"Clusters: {self.stats['num_clusters']}")
                stats_text.append(f"Target: {self.stats['target_cluster_size']} fish")
        
        for text in stats_text:
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
    
    def save_score(self):
        """Save simulation score to JSON file"""
        score_data = {
            "frame_count": self.frame_count,
            "boid_count": len(self.boids),
            "predator_count": len(self.predators),
            "statistics": self.stats,
            "config": CONFIG
        }
        
        try:
            with open(CONFIG["scoreOutputFile"], 'w') as f:
                json.dump(score_data, f, indent=4)
            print(f"Score saved to {CONFIG['scoreOutputFile']}")
        except Exception as e:
            print(f"Error saving score: {e}")
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_g:
                        CONFIG["visualizeGrid"] = not CONFIG["visualizeGrid"]
                    elif event.key == pygame.K_v:
                        CONFIG["visualizationMode"] = (CONFIG["visualizationMode"] + 1) % 3
                    elif event.key == pygame.K_c:
                        # Toggle cooperative hunting
                        CONFIG["cooperativeHunting"] = not CONFIG["cooperativeHunting"]
                        if CONFIG["cooperativeHunting"] and len(self.predators) > 1:
                            self.hunting_pack = HuntingPack(self.predators)
                        else:
                            self.hunting_pack = None
                        print(f"Cooperative hunting: {'ON' if CONFIG['cooperativeHunting'] else 'OFF'}")
                    elif event.key == pygame.K_l:
                        # Toggle clustering
                        CONFIG["enableClustering"] = not CONFIG["enableClustering"]
                        print(f"Smart clustering: {'ON' if CONFIG['enableClustering'] else 'OFF'}")
                    elif event.key == pygame.K_t:
                        # Cycle target strategy
                        strategies = ["nearest", "largest", "densest"]
                        current_idx = strategies.index(CONFIG["targetStrategy"])
                        next_idx = (current_idx + 1) % len(strategies)
                        CONFIG["targetStrategy"] = strategies[next_idx]
                        print(f"Target strategy: {CONFIG['targetStrategy'].upper()}")
                    elif event.key == pygame.K_SPACE:
                        self.save_score()
            
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        self.save_score()
        pygame.quit()
        sys.exit()

# ==================== MAIN ====================
if __name__ == "__main__":
    print("Advanced Boids Simulation with Cooperative Hunting & Smart Clustering")
    print("\nControls:")
    print("  ESC   - Quit")
    print("  G     - Toggle grid visualization")
    print("  V     - Cycle visualization modes (0=normal, 1=grid, 2=debug)")
    print("  C     - Toggle cooperative hunting ON/OFF")
    print("  L     - Toggle smart clustering ON/OFF")
    print("  T     - Cycle target strategy (nearest/largest/densest)")
    print("  SPACE - Save score to JSON")
    print("\nCooperative Hunting:")
    print("  When enabled, predators work together to encircle prey")
    print("  They position themselves around the group before bursting")
    print("  Watch the 'Formation' stat to see coordination quality")
    print("\nSmart Clustering:")
    print("  Detects actual groups of fish instead of using global centroid")
    print("  Prevents predators from targeting empty space between split groups")
    print("  Press V twice to see clusters visualized (green = target)")
    print(f"\nCurrent Settings:")
    print(f"  Cooperative Hunting: {'ON' if CONFIG['cooperativeHunting'] else 'OFF'}")
    print(f"  Smart Clustering: {'ON' if CONFIG['enableClustering'] else 'OFF'}")
    print(f"  Target Strategy: {CONFIG['targetStrategy'].upper()}")
    print("\nStarting simulation...")
    
    sim = Simulation()
    sim.run()
