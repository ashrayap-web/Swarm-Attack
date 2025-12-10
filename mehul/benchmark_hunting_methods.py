import pygame
import random
import math
import json
import time
from collections import defaultdict
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

# This tells pygame to use a non-existent video device
# Comment this out when recording videos
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ==================== SIMULATION PARAMETERS ====================
BASE_CONFIG = {
    "screenWidth": 1200,
    "screenHeight": 650,
    "boidCount": 100,
    "predatorCount": 5,
    "maxSpeed": 4.0,
    "maxForce": 0.1,
    "neighborRadius": 50,
    "desiredSeparation": 20,
    "predatorAvoidRadius": 80,
    "catchingDistance": 8,
    "gridCellSize": 80,
    "fpsTarget": 60,
    "speedMultiplier": 1.0,
    "backgroundColor": [25, 25, 25],
    "boidColor": [200, 200, 255],
    "predatorColor": [255, 50, 50],
    "maxAgentCount": 250,
    "accelerationFactor": 1.0,
    "movementNoise": 0.01,
    "adaptiveBehaviorEnabled": True,
    "learningRate": 0.05,
    "splitMergeEnabled": False,  # Disabled for consistent benchmarking
    
    # Shark/Predator herding parameters
    "predatorPerception": 180,
    "K_C": 0.2,
    "K_D": 0.1,
    "K_B": 0.3,
    "preyDensityThreshold": 20,
    "burstDuration": 1000,
    "cruiseSpeed": 2,
    "burstSpeed": 4.0,
    "maxAngularVelocitySlow": 0.12,
    "maxAngularVelocityFast": 0.05,
    "minSpeedForTurning": 0.1,
    
    # Cooperative hunting parameters
    "formationRadius": 150,
    "formationQualityThreshold": 0.5,
    "K_FORMATION": 0.25,
    "K_SEPARATION": 0.03,
    "minPredatorSeparation": 80,
    "compressionForce": 0.15,
    
    # Smart clustering parameters
    "clusterRadius": 60,
    "minClusterSize": 5,
    "targetStrategy": "nearest",
    "maxTargetDistance": 400,
}

# Benchmark configuration
BENCHMARK_DURATION = 5000  # frames (about 2.8 minutes at 60 FPS)
NUM_TRIALS = 50  # Number of trials per configuration
RECORD_VIDEO = False  # Record video for one trial of each method
VIDEO_TRIAL = 1  # Which trial to record (1-indexed)
VIDEO_FPS = 60  # FPS for recorded video (lower than simulation for smaller files)

WIDTH = BASE_CONFIG["screenWidth"]
HEIGHT = BASE_CONFIG["screenHeight"]
GRID_SIZE = BASE_CONFIG["gridCellSize"]

# Rule weights
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.3
COHESION_WEIGHT = 1.0
BOUNDARY_WEIGHT = 4
PREDATOR_AVOID_WEIGHT = 3.0
BOUNDARY_MARGIN = 80

# ==================== SPATIAL GRID ====================
class SpatialGrid:
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
    def __init__(self, x, y, max_speed, max_force, config):
        self.position = pygame.Vector2(x, y)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1.0, max_speed)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = max_speed * config["speedMultiplier"]
        self.max_force = max_force * config["accelerationFactor"]
        self.config = config
    
    def apply_force(self, force):
        self.acceleration += force
    
    def update(self):
        if self.config["movementNoise"] > 0:
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
    
    def steering(self, desired):
        steer = desired - self.velocity
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

# ==================== BOID CLASS ====================
class Boid(Agent):
    def __init__(self, x, y, config):
        super().__init__(x, y, config["maxSpeed"], config["maxForce"], config)
        self.stress_level = 0.0
        self.adaptive_weights = {
            "separation": SEPARATION_WEIGHT,
            "alignment": ALIGNMENT_WEIGHT,
            "cohesion": COHESION_WEIGHT
        }
    
    def flock(self, spatial_grid, predators):
        neighbors = spatial_grid.get_neighbors(self.position, self.config["neighborRadius"])
        neighbors = [n for n in neighbors if isinstance(n, Boid) and n is not self]
        
        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        boundary = self.boundary_repulsion()
        pred_avoid = self.avoid_predators(predators)
        
        if self.config["adaptiveBehaviorEnabled"]:
            self.adapt_behavior(neighbors, predators)
            sep_w = self.adaptive_weights["separation"]
            ali_w = self.adaptive_weights["alignment"]
            coh_w = self.adaptive_weights["cohesion"]
        else:
            sep_w = SEPARATION_WEIGHT
            ali_w = ALIGNMENT_WEIGHT
            coh_w = COHESION_WEIGHT
        
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
            if 0 < dist < self.config["desiredSeparation"]:
                diff = self.position - other.position
                diff /= dist
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
        
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def avoid_predators(self, predators):
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
                steering.scale_to_length(self.max_speed * 1.5)
                steering = self.steering(steering)
        else:
            self.stress_level = max(0.0, self.stress_level - 0.02)
        
        return steering
    
    def adapt_behavior(self, neighbors, predators):
        if not self.config["adaptiveBehaviorEnabled"]:
            return
        
        lr = self.config["learningRate"]
        
        if self.stress_level > 0.5:
            self.adaptive_weights["separation"] += lr * 0.5
            self.adaptive_weights["cohesion"] -= lr * 0.2
        else:
            self.adaptive_weights["separation"] += lr * (SEPARATION_WEIGHT - self.adaptive_weights["separation"])
            self.adaptive_weights["alignment"] += lr * (ALIGNMENT_WEIGHT - self.adaptive_weights["alignment"])
            self.adaptive_weights["cohesion"] += lr * (COHESION_WEIGHT - self.adaptive_weights["cohesion"])
        
        for key in self.adaptive_weights:
            self.adaptive_weights[key] = max(0.1, min(3.0, self.adaptive_weights[key]))

# ==================== PREY CLUSTERING ====================
class PreyCluster:
    def __init__(self, fish_list):
        self.fish = fish_list
        self.centroid = self._calculate_centroid()
        self.size = len(fish_list)
        self.radius = self._calculate_radius()
        self.density = self._calculate_density()
    
    def _calculate_centroid(self):
        if not self.fish:
            return pygame.Vector2(0, 0)
        center = pygame.Vector2(0, 0)
        for fish in self.fish:
            center += fish.position
        return center / len(self.fish)
    
    def _calculate_radius(self):
        if not self.fish:
            return 0
        max_dist = 0
        for fish in self.fish:
            dist = self.centroid.distance_to(fish.position)
            if dist > max_dist:
                max_dist = dist
        return max_dist
    
    def _calculate_density(self):
        if self.radius == 0:
            return float('inf') if self.size > 0 else 0
        area = math.pi * self.radius * self.radius
        return self.size / area if area > 0 else 0
    
    def distance_to(self, position):
        return self.centroid.distance_to(position)

def detect_prey_clusters(boids, config):
    if not boids or not config["enableClustering"]:
        return [PreyCluster(boids)] if boids else []
    
    cluster_radius = config["clusterRadius"]
    min_cluster_size = config["minClusterSize"]
    
    unassigned = set(boids)
    clusters = []
    
    while unassigned:
        seed = unassigned.pop()
        current_cluster = [seed]
        to_check = [seed]
        
        while to_check:
            checking_boid = to_check.pop(0)
            neighbors = []
            for boid in list(unassigned):
                dist = checking_boid.position.distance_to(boid.position)
                if dist < cluster_radius:
                    neighbors.append(boid)
            
            for neighbor in neighbors:
                if neighbor in unassigned:
                    current_cluster.append(neighbor)
                    unassigned.remove(neighbor)
                    to_check.append(neighbor)
        
        if len(current_cluster) >= min_cluster_size:
            clusters.append(PreyCluster(current_cluster))
    
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters

# ==================== HUNTING PACK CLASS ====================
class HuntingPack:
    def __init__(self, predators, config):
        self.predators = predators
        self.config = config
        self.collective_state = "herding"
        self.burst_start_time = None
        self.prey_centroid = None
        self.formation_quality = 0.0
        self.assigned_angles = {}
        self.target_cluster = None
        self.all_clusters = []
    
    def update_collective_state(self, boids, current_time):
        if not boids or not self.predators:
            return
        
        if self.config["enableClustering"]:
            self.all_clusters = detect_prey_clusters(boids, self.config)
            self.target_cluster = self._select_target_cluster()
            
            if self.target_cluster:
                self.prey_centroid = self.target_cluster.centroid
            else:
                self.prey_centroid = pygame.Vector2(0, 0)
                for boid in boids:
                    self.prey_centroid += boid.position
                self.prey_centroid /= len(boids)
        else:
            self.prey_centroid = pygame.Vector2(0, 0)
            for boid in boids:
                self.prey_centroid += boid.position
            self.prey_centroid /= len(boids)
            self.target_cluster = PreyCluster(boids)
            self.all_clusters = [self.target_cluster]
        
        self._assign_formation_positions()
        self.formation_quality = self._calculate_formation_quality()
        
        if self.target_cluster:
            prey_density = self.target_cluster.size
        else:
            prey_density = self._calculate_global_prey_density(boids)
        
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
    
    def _select_target_cluster(self):
        if not self.all_clusters:
            return None
        
        valid_clusters = []
        max_dist = self.config["maxTargetDistance"]
        
        for cluster in self.all_clusters:
            for pred in self.predators:
                if cluster.distance_to(pred.position) < max_dist:
                    valid_clusters.append(cluster)
                    break
        
        if not valid_clusters:
            valid_clusters = self.all_clusters
        
        strategy = self.config["targetStrategy"]
        
        if strategy == "largest":
            return max(valid_clusters, key=lambda c: c.size)
        elif strategy == "densest":
            return max(valid_clusters, key=lambda c: c.density)
        else:
            pack_center = pygame.Vector2(0, 0)
            for pred in self.predators:
                pack_center += pred.position
            pack_center /= len(self.predators)
            return min(valid_clusters, key=lambda c: c.distance_to(pack_center))
    
    def _assign_formation_positions(self):
        if not self.predators:
            return
        
        num_predators = len(self.predators)
        angle_spacing = (2 * math.pi) / num_predators
        
        predators_with_angles = []
        for pred in self.predators:
            to_prey = self.prey_centroid - pred.position
            if to_prey.length() > 0.001:
                current_angle = math.atan2(to_prey.y, to_prey.x)
            else:
                current_angle = 0
            predators_with_angles.append((pred, current_angle))
        
        predators_with_angles.sort(key=lambda x: x[1])
        
        for i, (pred, _) in enumerate(predators_with_angles):
            target_angle = i * angle_spacing
            self.assigned_angles[pred] = target_angle
    
    def _calculate_formation_quality(self):
        if not self.predators or len(self.predators) < 2:
            return 1.0
        
        ideal_spacing = (2 * math.pi) / len(self.predators)
        
        actual_angles = []
        for pred in self.predators:
            to_prey = self.prey_centroid - pred.position
            if to_prey.length() > 0.001:
                angle = math.atan2(to_prey.y, to_prey.x)
                actual_angles.append(angle)
        
        if not actual_angles:
            return 0.0
        
        actual_angles.sort()
        
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
        
        distance_quality = 0.0
        for pred in self.predators:
            dist = pred.position.distance_to(self.prey_centroid)
            ideal_dist = self.config['formationRadius']
            dist_error = abs(dist - ideal_dist) / ideal_dist
            distance_quality += max(0, 1.0 - dist_error)
        distance_quality /= len(self.predators)
        
        return (quality * 0.6 + distance_quality * 0.4)
    
    def _calculate_global_prey_density(self, boids):
        if not self.prey_centroid:
            return 0
        
        count = 0
        radius = self.config['predatorPerception']
        for boid in boids:
            dist = self.prey_centroid.distance_to(boid.position)
            if dist < radius:
                count += 1
        
        return count
    
    def get_formation_position(self, predator):
        if predator not in self.assigned_angles or not self.prey_centroid:
            return None
        
        target_angle = self.assigned_angles[predator]
        formation_radius = self.config['formationRadius']
        
        target_x = self.prey_centroid.x + formation_radius * math.cos(target_angle)
        target_y = self.prey_centroid.y + formation_radius * math.sin(target_angle)
        
        return pygame.Vector2(target_x, target_y)

# ==================== PREDATOR CLASS ====================
class Predator(Agent):
    def __init__(self, x, y, config):
        super().__init__(x, y, config["cruiseSpeed"], config["maxForce"] * 1.2, config)
        self.state = "herding"
        self.burst_start_time = None
        self.perception = config["predatorPerception"]
        self.distance_traveled = 0.0
        self.previous_position = self.position.copy()
        
        if self.velocity.length() > 0:
            self.previous_direction = math.atan2(self.velocity.y, self.velocity.x)
        else:
            self.previous_direction = 0.0
    
    def calculate_prey_centroid(self, boids):
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
        count = 0
        perception_sq = self.perception * self.perception
        
        for boid in boids:
            dist_sq = self.position.distance_squared_to(boid.position)
            if dist_sq < perception_sq:
                count += 1
        
        return count
    
    def herding_acceleration(self, prey_centroid):
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        centering_force = prey_centroid - self.position
        centering_force = centering_force * self.config['K_C']
        damping_force = -self.velocity * self.config['K_D']
        acceleration = centering_force + damping_force
        return acceleration
    
    def calculate_interception(self, target):
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
        predicted_position = target.position + (target.velocity * time_to_intercept)
        
        return predicted_position
    
    def burst_interception(self, boids):
        if not boids:
            return pygame.Vector2(0, 0)
        
        closest_prey = min(boids, key=lambda b: self.position.distance_to(b.position))
        
        if self.config["predatorInterception"]:
            target_pos = self.calculate_interception(closest_prey)
        else:
            target_pos = closest_prey.position
        
        direction = target_pos - self.position
        if direction.length() > 0.001:
            acceleration = direction * self.config['K_B']
            return acceleration
        
        return pygame.Vector2(0, 0)
    
    def constrain_turning(self, desired_velocity):
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
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        speed_ratio = current_speed / max(self.max_speed, 0.001)
        speed_ratio = max(0.0, min(1.0, speed_ratio))
        max_angular_velocity = self.config['maxAngularVelocitySlow'] * (1 - speed_ratio) + self.config['maxAngularVelocityFast'] * speed_ratio
        
        if abs(angle_diff) > max_angular_velocity:
            angle_diff = math.copysign(max_angular_velocity, angle_diff)
        
        constrained_direction = current_direction + angle_diff
        constrained_velocity = pygame.Vector2(
            math.cos(constrained_direction) * desired_speed,
            math.sin(constrained_direction) * desired_speed
        )
        
        return constrained_velocity
    
    def update_state(self, boids, current_time):
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
    
    def cooperative_hunt(self, boids, other_predators, hunting_pack):
        caught_boid = None
        
        if not boids:
            return caught_boid
        
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
        
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < self.config["catchingDistance"]:
                caught_boid = boid
                break
        
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def hunt(self, boids):
        caught_boid = None
        
        if not boids:
            return caught_boid
        
        current_time = pygame.time.get_ticks()
        self.update_state(boids, current_time)
        
        prey_centroid, prey_count = self.calculate_prey_centroid(boids)
        
        if self.state == "herding":
            if prey_centroid is not None:
                acceleration = self.herding_acceleration(prey_centroid)
                self.apply_force(acceleration)
        elif self.state == "burst":
            acceleration = self.burst_interception(boids)
            self.apply_force(acceleration)
        
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < self.config["catchingDistance"]:
                caught_boid = boid
                break
        
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def formation_acceleration(self, formation_pos, prey_centroid):
        to_formation = formation_pos - self.position
        formation_force = to_formation * self.config['K_FORMATION']
        
        to_prey = prey_centroid - self.position
        if to_prey.length() > 0.001:
            tangent = pygame.Vector2(-to_prey.y, to_prey.x)
            tangent = tangent.normalize() * self.config['K_D']
            formation_force += tangent
        
        return formation_force
    
    def compression_acceleration(self, prey_centroid):
        to_prey = prey_centroid - self.position
        dist = to_prey.length()
        
        if dist > self.config['formationRadius']:
            compression = to_prey.normalize() * self.config['compressionForce']
            return compression
        
        return pygame.Vector2(0, 0)
    
    def predator_separation(self, other_predators):
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
        if self.config["movementNoise"] > 0:
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

# ==================== BENCHMARK SIMULATION ====================
class BenchmarkSimulation:
    def __init__(self, config, enable_video=False, video_filename=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.config = config
        
        # Video recording setup
        self.enable_video = enable_video
        self.video_writer = None
        self.video_filename = video_filename
        self.frame_skip = max(1, BASE_CONFIG["fpsTarget"] // VIDEO_FPS)  # Sample frames for video
        
        if self.enable_video and self.video_filename:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                fourcc,
                VIDEO_FPS,
                (WIDTH, HEIGHT)
            )
            print(f"  Recording video to: {self.video_filename}")
        
        self.spatial_grid = SpatialGrid(WIDTH, HEIGHT, GRID_SIZE)
        self.boids = []
        self.predators = []
        
        self._spawn_agents()
        
        if config["cooperativeHunting"] and len(self.predators) > 1:
            self.hunting_pack = HuntingPack(self.predators, config)
        else:
            self.hunting_pack = None
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # Enhanced statistics
        self.stats = {
            "boids_eaten": 0,
            "total_distance_traveled": 0,
            "burst_attempts": 0,
            "successful_bursts": 0,
            "burst_captures": 0,
            "herding_captures": 0,
            "time_in_burst": 0,
            "time_in_herding": 0,
            "avg_formation_quality": 0,
            "formation_quality_samples": 0,
            "avg_stress": 0,
            "avg_cohesion": 0,
            "avg_predator_speed": 0,
            "predator_speed_samples": 0,
            "captures_per_1000_frames": [],
            "first_capture_frame": None,
            "time_between_captures": [],
            "last_capture_frame": 0,
            "cumulative_captures_over_time": [],  # Track captures at each frame
        }
        
        self.previous_pack_state = None
    
    def _spawn_agents(self):
        for _ in range(self.config["boidCount"]):
            x = random.uniform(100, WIDTH - 100)
            y = random.uniform(100, HEIGHT - 100)
            self.boids.append(Boid(x, y, self.config))
        
        for _ in range(self.config["predatorCount"]):
            x = random.uniform(50, WIDTH - 50)
            y = random.uniform(50, HEIGHT - 50)
            self.predators.append(Predator(x, y, self.config))
    
    def update(self):
        self.spatial_grid.clear()
        for boid in self.boids:
            self.spatial_grid.insert(boid)
        
        for boid in self.boids:
            boid.flock(self.spatial_grid, self.predators)
            boid.update()
        
        if self.hunting_pack and self.config["cooperativeHunting"]:
            current_time = pygame.time.get_ticks()
            self.hunting_pack.update_collective_state(self.boids, current_time)
            
            # Track burst attempts
            if self.previous_pack_state == "herding" and self.hunting_pack.collective_state == "burst":
                self.stats["burst_attempts"] += 1
            
            self.previous_pack_state = self.hunting_pack.collective_state
        
        caught_boids = []
        for predator in self.predators:
            if self.hunting_pack and self.config["cooperativeHunting"]:
                caught = predator.cooperative_hunt(self.boids, self.predators, self.hunting_pack)
            else:
                caught = predator.hunt(self.boids)
            
            if caught:
                caught_boids.append(caught)
                
                # Track which state the capture happened in
                if predator.state == "burst":
                    self.stats["burst_captures"] += 1
                else:
                    self.stats["herding_captures"] += 1
            
            predator.update()
        
        # Remove caught boids and track timing
        for caught in caught_boids:
            if caught in self.boids:
                self.boids.remove(caught)
                self.stats["boids_eaten"] += 1
                
                if self.stats["first_capture_frame"] is None:
                    self.stats["first_capture_frame"] = self.frame_count
                
                frames_since_last = self.frame_count - self.stats["last_capture_frame"]
                if self.stats["last_capture_frame"] > 0:
                    self.stats["time_between_captures"].append(frames_since_last)
                
                self.stats["last_capture_frame"] = self.frame_count
        
        # Track successful bursts (burst that resulted in capture)
        if caught_boids and self.hunting_pack and self.hunting_pack.collective_state == "burst":
            self.stats["successful_bursts"] += 1
        
        self.frame_count += 1
        self._update_statistics()
    
    def _update_statistics(self):
        # Track distance traveled
        total_dist = sum(pred.distance_traveled for pred in self.predators)
        self.stats["total_distance_traveled"] = total_dist
        
        # Track time in each state
        if self.hunting_pack:
            if self.hunting_pack.collective_state == "burst":
                self.stats["time_in_burst"] += 1
            else:
                self.stats["time_in_herding"] += 1
            
            # Track formation quality
            self.stats["avg_formation_quality"] += self.hunting_pack.formation_quality
            self.stats["formation_quality_samples"] += 1
        
        # Track predator speed
        if self.predators:
            total_speed = sum(pred.velocity.length() for pred in self.predators)
            self.stats["avg_predator_speed"] += total_speed / len(self.predators)
            self.stats["predator_speed_samples"] += 1
        
        # Track prey behavior
        if self.boids:
            total_stress = sum(b.stress_level for b in self.boids)
            self.stats["avg_stress"] = total_stress / len(self.boids)
            
            centroid = pygame.Vector2(0, 0)
            for b in self.boids:
                centroid += b.position
            centroid /= len(self.boids)
            
            total_dist = sum(b.position.distance_to(centroid) for b in self.boids)
            self.stats["avg_cohesion"] = total_dist / len(self.boids)
        
        # Track captures per 1000 frames
        if self.frame_count % 1000 == 0:
            recent_captures = sum(1 for t in self.stats["time_between_captures"] 
                                if self.frame_count - t <= 1000)
            self.stats["captures_per_1000_frames"].append(recent_captures)
        
        # Track cumulative captures at regular intervals for plotting
        if self.frame_count % 10 == 0:  # Sample every 10 frames
            self.stats["cumulative_captures_over_time"].append({
                "frame": self.frame_count,
                "captures": self.stats["boids_eaten"]
            })
    
    def run_benchmark(self, max_frames):
        print(f"Running benchmark for {max_frames} frames...")
        
        while self.frame_count < max_frames:
            self.update()
            
            # Render and capture frame for video if enabled
            if self.enable_video and self.video_writer:
                if self.frame_count % self.frame_skip == 0:
                    self._render_frame()
                    self._capture_frame()
            
            # Progress indicator
            if self.frame_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                progress = (self.frame_count / max_frames) * 100
                print(f"  Progress: {progress:.1f}% ({self.frame_count}/{max_frames} frames, "
                      f"{elapsed:.1f}s elapsed, {self.stats['boids_eaten']} captures)")
        
        # Cleanup video writer
        if self.video_writer:
            self.video_writer.release()
            print(f"  Video saved successfully!")
        
        pygame.quit()
        return self.get_results()
    
    def _render_frame(self):
        """Render the current frame with debug visualization"""
        self.screen.fill(self.config["backgroundColor"])
        
        debug_mode = 2  # Full debug mode for video
        
        # Draw cooperative hunting visualization
        if debug_mode >= 2 and self.hunting_pack:
            self._draw_hunting_visualization()
        
        # Draw boids
        for boid in self.boids:
            self._draw_boid(boid, debug_mode)
        
        # Draw predators with formation positions
        for predator in self.predators:
            formation_pos = None
            if self.hunting_pack and debug_mode >= 2:
                formation_pos = self.hunting_pack.get_formation_position(predator)
            self._draw_predator(predator, debug_mode, formation_pos)
        
        # Draw stats overlay
        self._draw_stats()
        
        pygame.display.flip()
    
    def _draw_boid(self, boid, debug_mode=False):
        """Draw a single boid"""
        color = self.config["boidColor"]
        if debug_mode == 2:
            stress_color = int(255 * boid.stress_level)
            color = (stress_color, color[1], color[2])
        
        pygame.draw.circle(self.screen, color, (int(boid.position.x), int(boid.position.y)), 4)
        
        if debug_mode >= 1 and boid.velocity.length() > 0:
            end_pos = boid.position + boid.velocity.normalize() * 10
            pygame.draw.line(self.screen, color, boid.position, end_pos, 1)
    
    def _draw_predator(self, predator, debug_mode=False, formation_pos=None):
        """Draw a single predator"""
        # Change color based on state
        if predator.state == "burst":
            draw_color = (255, 100, 100)
        else:
            draw_color = self.config["predatorColor"]
        
        pygame.draw.circle(self.screen, draw_color, (int(predator.position.x), int(predator.position.y)), 6)
        pygame.draw.circle(self.screen, (200, 0, 0), (int(predator.position.x), int(predator.position.y)), 6, 2)
        
        # Debug mode visualizations
        if debug_mode >= 2:
            # Perception radius
            pygame.draw.circle(self.screen, (100, 50, 50), 
                             (int(predator.position.x), int(predator.position.y)), 
                             int(predator.perception), 1)
            
            # Formation target during herding
            if formation_pos and predator.state == "herding":
                pygame.draw.circle(self.screen, (255, 255, 0), 
                                 (int(formation_pos.x), int(formation_pos.y)), 5, 2)
                pygame.draw.line(self.screen, (255, 255, 0), predator.position, formation_pos, 1)
            
            # Interception point during burst
            if predator.state == "burst" and self.boids and self.config["predatorInterception"]:
                closest_prey = min(self.boids, key=lambda b: predator.position.distance_to(b.position))
                intercept_pos = predator.calculate_interception(closest_prey)
                pygame.draw.line(self.screen, (255, 150, 0), predator.position, intercept_pos, 2)
                size = 5
                pygame.draw.line(self.screen, (255, 200, 0), 
                               (intercept_pos.x - size, intercept_pos.y - size),
                               (intercept_pos.x + size, intercept_pos.y + size), 2)
                pygame.draw.line(self.screen, (255, 200, 0),
                               (intercept_pos.x + size, intercept_pos.y - size),
                               (intercept_pos.x - size, intercept_pos.y + size), 2)
    
    def _draw_hunting_visualization(self):
        """Draw visualization of cooperative hunting behavior"""
        if not self.hunting_pack:
            return
        
        # Draw all detected clusters
        if self.config["enableClustering"] and self.hunting_pack.all_clusters:
            for cluster in self.hunting_pack.all_clusters:
                color = (80, 80, 80)
                if cluster == self.hunting_pack.target_cluster:
                    color = (100, 200, 100)
                
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
        
        # Draw target prey centroid
        if self.hunting_pack.prey_centroid:
            centroid = self.hunting_pack.prey_centroid
            pygame.draw.circle(self.screen, (100, 255, 100), 
                             (int(centroid.x), int(centroid.y)), 10, 3)
            
            # Draw formation circle
            pygame.draw.circle(self.screen, (150, 200, 50), 
                             (int(centroid.x), int(centroid.y)), 
                             int(self.config['formationRadius']), 2)
    
    def _draw_stats(self):
        """Draw statistics overlay on screen"""
        font = pygame.font.Font(None, 24)
        y_offset = 10
        
        stats_text = [
            f"Frame: {self.frame_count}/{BENCHMARK_DURATION}",
            f"Boids: {len(self.boids)}",
            f"Eaten: {self.stats['boids_eaten']}",
            f"Avg Stress: {self.stats['avg_stress']:.2f}",
            f"Cohesion: {self.stats['avg_cohesion']:.1f}"
        ]
        
        # Add cooperative hunting stats
        if self.config["cooperativeHunting"] and self.hunting_pack:
            stats_text.append(f"Formation: {self.hunting_pack.formation_quality:.2f}")
            stats_text.append(f"Pack: {self.hunting_pack.collective_state}")
            if self.config["enableClustering"]:
                stats_text.append(f"Clusters: {len(self.hunting_pack.all_clusters)}")
                if self.hunting_pack.target_cluster:
                    stats_text.append(f"Target: {self.hunting_pack.target_cluster.size} fish")
        
        for text in stats_text:
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
    
    def _capture_frame(self):
        """Capture the current pygame screen to video"""
        if not self.video_writer:
            return
        
        # Get the pygame surface as a numpy array
        frame = pygame.surfarray.array3d(self.screen)
        
        # Transpose to get correct orientation (pygame uses (width, height, channels))
        frame = np.transpose(frame, (1, 0, 2))
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        self.video_writer.write(frame)
    
    def get_results(self):
        elapsed_time = time.time() - self.start_time
        
        # Calculate derived metrics
        avg_formation = 0
        if self.stats["formation_quality_samples"] > 0:
            avg_formation = self.stats["avg_formation_quality"] / self.stats["formation_quality_samples"]
        
        avg_time_between = 0
        if self.stats["time_between_captures"]:
            avg_time_between = sum(self.stats["time_between_captures"]) / len(self.stats["time_between_captures"])
        
        burst_success_rate = 0
        if self.stats["burst_attempts"] > 0:
            burst_success_rate = self.stats["successful_bursts"] / self.stats["burst_attempts"]
        
        captures_per_distance = 0
        if self.stats["total_distance_traveled"] > 0:
            captures_per_distance = self.stats["boids_eaten"] / self.stats["total_distance_traveled"]
        
        captures_per_frame = 0
        if self.frame_count > 0:
            captures_per_frame = self.stats["boids_eaten"] / self.frame_count
        
        avg_predator_speed = 0
        if self.stats["predator_speed_samples"] > 0:
            avg_predator_speed = self.stats["avg_predator_speed"] / self.stats["predator_speed_samples"]
        
        return {
            "total_captures": self.stats["boids_eaten"],
            "frames": self.frame_count,
            "elapsed_time_seconds": elapsed_time,
            "total_distance_traveled": self.stats["total_distance_traveled"],
            "burst_captures": self.stats["burst_captures"],
            "herding_captures": self.stats["herding_captures"],
            "burst_attempts": self.stats["burst_attempts"],
            "successful_bursts": self.stats["successful_bursts"],
            "burst_success_rate": burst_success_rate,
            "time_in_burst": self.stats["time_in_burst"],
            "time_in_herding": self.stats["time_in_herding"],
            "avg_formation_quality": avg_formation,
            "avg_stress": self.stats["avg_stress"],
            "avg_cohesion": self.stats["avg_cohesion"],
            "avg_predator_speed": avg_predator_speed,
            "first_capture_frame": self.stats["first_capture_frame"],
            "avg_time_between_captures": avg_time_between,
            "captures_per_distance": captures_per_distance,
            "captures_per_frame": captures_per_frame,
            "captures_per_1000_frames": self.stats["captures_per_1000_frames"],
            "final_prey_count": len(self.boids),
            "cumulative_captures_over_time": self.stats["cumulative_captures_over_time"],
        }

# ==================== MAIN BENCHMARK ====================
def run_configuration_benchmark(config_name, config_overrides, num_trials, duration, method_id):
    """Run multiple trials of a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    trial_results = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Create config for this trial
        config = BASE_CONFIG.copy()
        config.update(config_overrides)
        
        # Set random seed for reproducibility
        random.seed(42 + trial)
        
        # Determine if this trial should be recorded
        enable_video = RECORD_VIDEO and (trial + 1) == VIDEO_TRIAL
        video_filename = None
        
        if enable_video:
            # Create clean filename from config name
            safe_name = config_name.lower().replace(" ", "_").replace(":", "").replace("+", "and")
            video_filename = f"recording_method{method_id}_{safe_name}_trial{trial + 1}.mp4"
        
        # Run simulation
        sim = BenchmarkSimulation(config, enable_video=enable_video, video_filename=video_filename)
        results = sim.run_benchmark(duration)
        results["trial"] = trial + 1
        if enable_video:
            results["video_file"] = video_filename
        trial_results.append(results)
    
    return trial_results

def calculate_aggregate_stats(trial_results):
    """Calculate mean and std dev across trials"""
    if not trial_results:
        return {}
    
    # Metrics to aggregate
    metrics = [
        "total_captures", "total_distance_traveled", "captures_per_distance",
        "captures_per_frame", "burst_success_rate", "avg_formation_quality",
        "avg_stress", "avg_cohesion", "avg_predator_speed", "first_capture_frame", 
        "avg_time_between_captures", "burst_captures", "herding_captures", "elapsed_time_seconds"
    ]
    
    aggregates = {}
    
    for metric in metrics:
        values = [r[metric] for r in trial_results if metric in r and r[metric] is not None]
        if values:
            aggregates[f"{metric}_mean"] = sum(values) / len(values)
            if len(values) > 1:
                mean = aggregates[f"{metric}_mean"]
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                aggregates[f"{metric}_std"] = math.sqrt(variance)
            else:
                aggregates[f"{metric}_std"] = 0
    
    return aggregates

def plot_cumulative_captures(results1, results2, results3):
    """Plot cumulative captures over time for all three methods"""
    plt.figure(figsize=(12, 7))
    
    # Use first trial from each method for the plot
    trial1_data = results1[0]["cumulative_captures_over_time"]
    trial2_data = results2[0]["cumulative_captures_over_time"]
    trial3_data = results3[0]["cumulative_captures_over_time"]
    
    # Extract frames and captures
    frames1 = [d["frame"] for d in trial1_data]
    captures1 = [d["captures"] for d in trial1_data]
    
    frames2 = [d["frame"] for d in trial2_data]
    captures2 = [d["captures"] for d in trial2_data]
    
    frames3 = [d["frame"] for d in trial3_data]
    captures3 = [d["captures"] for d in trial3_data]
    
    # Plot lines
    plt.plot(frames1, captures1, label='Basic Herding', linewidth=2, color='#FF6B6B')
    plt.plot(frames2, captures2, label='Herding + Interception', linewidth=2, color='#4ECDC4')
    plt.plot(frames3, captures3, label='Full Coordination', linewidth=2, color='#95E1D3')
    
    # Formatting
    plt.xlabel('Frame Number', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Fish Captured', fontsize=12, fontweight='bold')
    plt.title('Hunting Performance Comparison: Cumulative Captures Over Time', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add final capture counts as annotations
    plt.text(frames1[-1], captures1[-1], f' {captures1[-1]}', 
             verticalalignment='center', fontsize=9, color='#FF6B6B')
    plt.text(frames2[-1], captures2[-1], f' {captures2[-1]}', 
             verticalalignment='center', fontsize=9, color='#4ECDC4')
    plt.text(frames3[-1], captures3[-1], f' {captures3[-1]}', 
             verticalalignment='center', fontsize=9, color='#95E1D3')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "hunting_performance_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_filename}")
    
    # Display the plot
    plt.show()

def export_results_to_csv(results1, results2, results3, filename="benchmark_results.csv"):
    """Export benchmark results to CSV with specified format"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['simulation_id', 'boids_caught', 'avg_stress', 'avg_cohesion', 'avg_predator_speed', 'final_boid_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Write results for method 1
        for result in results1:
            writer.writerow({
                'simulation_id': f"method1_trial{result['trial']}",
                'boids_caught': result['total_captures'],
                'avg_stress': result['avg_stress'],
                'avg_cohesion': result['avg_cohesion'],
                'avg_predator_speed': result['avg_predator_speed'],
                'final_boid_count': result['final_prey_count']
            })
        
        # Write results for method 2
        for result in results2:
            writer.writerow({
                'simulation_id': f"method2_trial{result['trial']}",
                'boids_caught': result['total_captures'],
                'avg_stress': result['avg_stress'],
                'avg_cohesion': result['avg_cohesion'],
                'avg_predator_speed': result['avg_predator_speed'],
                'final_boid_count': result['final_prey_count']
            })
        
        # Write results for method 3
        for result in results3:
            writer.writerow({
                'simulation_id': f"method3_trial{result['trial']}",
                'boids_caught': result['total_captures'],
                'avg_stress': result['avg_stress'],
                'avg_cohesion': result['avg_cohesion'],
                'avg_predator_speed': result['avg_predator_speed'],
                'final_boid_count': result['final_prey_count']
            })
    
    print(f"\nCSV results saved to: {filename}")

def main():
    print("="*60)
    print("HUNTING METHOD BENCHMARK COMPARISON")
    print("="*60)
    print(f"Duration per trial: {BENCHMARK_DURATION} frames")
    print(f"Trials per configuration: {NUM_TRIALS}")
    print(f"Total simulations: {NUM_TRIALS * 3}")
    if RECORD_VIDEO:
        print(f"Video recording: ENABLED (trial {VIDEO_TRIAL} of each method)")
        print(f"Video FPS: {VIDEO_FPS}")
    print()
    
    # Configuration 1: Basic Herding
    config1 = {
        "cooperativeHunting": False,
        "predatorInterception": False,
        "enableClustering": False,
    }
    results1 = run_configuration_benchmark(
        "Method 1: Basic Herding",
        config1,
        NUM_TRIALS,
        BENCHMARK_DURATION,
        method_id=1
    )
    
    # Configuration 2: Herding + Interception
    config2 = {
        "cooperativeHunting": False,
        "predatorInterception": True,
        "enableClustering": False,
    }
    results2 = run_configuration_benchmark(
        "Method 2: Herding + Interception",
        config2,
        NUM_TRIALS,
        BENCHMARK_DURATION,
        method_id=2
    )
    
    # Configuration 3: Full Coordination
    config3 = {
        "cooperativeHunting": True,
        "predatorInterception": True,
        "enableClustering": True,
    }
    results3 = run_configuration_benchmark(
        "Method 3: Full Coordination (Clustering + Interception)",
        config3,
        NUM_TRIALS,
        BENCHMARK_DURATION,
        method_id=3
    )
    
    # Calculate aggregates
    agg1 = calculate_aggregate_stats(results1)
    agg2 = calculate_aggregate_stats(results2)
    agg3 = calculate_aggregate_stats(results3)
    
    # Prepare final report
    report = {
        "benchmark_config": {
            "duration_frames": BENCHMARK_DURATION,
            "trials_per_method": NUM_TRIALS,
            "base_config": BASE_CONFIG,
        },
        "method_1_basic_herding": {
            "config": config1,
            "trial_results": results1,
            "aggregates": agg1,
        },
        "method_2_herding_interception": {
            "config": config2,
            "trial_results": results2,
            "aggregates": agg2,
        },
        "method_3_full_coordination": {
            "config": config3,
            "trial_results": results3,
            "aggregates": agg3,
        },
        "comparison": {
            "winner_by_total_captures": None,
            "winner_by_efficiency": None,
            "winner_by_time_to_first": None,
        }
    }
    
    # Determine winners
    methods = [
        ("Basic Herding", agg1),
        ("Herding + Interception", agg2),
        ("Full Coordination", agg3)
    ]
    
    # Most captures
    best_captures = max(methods, key=lambda x: x[1].get("total_captures_mean", 0))
    report["comparison"]["winner_by_total_captures"] = best_captures[0]
    
    # Best efficiency (captures per distance)
    best_efficiency = max(methods, key=lambda x: x[1].get("captures_per_distance_mean", 0))
    report["comparison"]["winner_by_efficiency"] = best_efficiency[0]
    
    # Fastest first capture
    best_first = min(methods, key=lambda x: x[1].get("first_capture_frame_mean", float('inf')))
    report["comparison"]["winner_by_time_to_first"] = best_first[0]
    
    # Save results to JSON
    output_file = "hunting_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save results to CSV
    export_results_to_csv(results1, results2, results3)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print("\n1. BASIC HERDING:")
    print(f"   Total Captures: {agg1.get('total_captures_mean', 0):.2f} ± {agg1.get('total_captures_std', 0):.2f}")
    print(f"   Efficiency (captures/distance): {agg1.get('captures_per_distance_mean', 0):.6f}")
    print(f"   First Capture Frame: {agg1.get('first_capture_frame_mean', 0):.0f}")
    print(f"   Avg Time Between Captures: {agg1.get('avg_time_between_captures_mean', 0):.1f} frames")
    
    print("\n2. HERDING + INTERCEPTION:")
    print(f"   Total Captures: {agg2.get('total_captures_mean', 0):.2f} ± {agg2.get('total_captures_std', 0):.2f}")
    print(f"   Efficiency (captures/distance): {agg2.get('captures_per_distance_mean', 0):.6f}")
    print(f"   First Capture Frame: {agg2.get('first_capture_frame_mean', 0):.0f}")
    print(f"   Avg Time Between Captures: {agg2.get('avg_time_between_captures_mean', 0):.1f} frames")
    
    print("\n3. FULL COORDINATION:")
    print(f"   Total Captures: {agg3.get('total_captures_mean', 0):.2f} ± {agg3.get('total_captures_std', 0):.2f}")
    print(f"   Efficiency (captures/distance): {agg3.get('captures_per_distance_mean', 0):.6f}")
    print(f"   First Capture Frame: {agg3.get('first_capture_frame_mean', 0):.0f}")
    print(f"   Avg Time Between Captures: {agg3.get('avg_time_between_captures_mean', 0):.1f} frames")
    print(f"   Burst Success Rate: {agg3.get('burst_success_rate_mean', 0):.2%}")
    print(f"   Avg Formation Quality: {agg3.get('avg_formation_quality_mean', 0):.3f}")
    
    print("\n" + "="*60)
    print("WINNERS:")
    print("="*60)
    print(f"Most Captures: {report['comparison']['winner_by_total_captures']}")
    print(f"Best Efficiency: {report['comparison']['winner_by_efficiency']}")
    print(f"Fastest First Capture: {report['comparison']['winner_by_time_to_first']}")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("="*60)
    
    # Generate and display comparison plot
    print("\nGenerating comparison plot...")
    plot_cumulative_captures(results1, results2, results3)

if __name__ == "__main__":
    main()
