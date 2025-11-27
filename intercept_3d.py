import pygame
import random
import math
import sys
import json
from collections import defaultdict

# ==================== SIMULATION PARAMETERS ====================
CONFIG = {
    "screenWidth": 1200,
    "screenHeight": 800,
    "depth": 1000,  # Z-axis depth
    "boidCount": 150,
    "predatorCount": 5,
    "maxSpeed": 5.0,
    "maxForce": 0.15,
    "neighborRadius": 100,
    "desiredSeparation": 30,
    "predatorAvoidRadius": 120,
    "catchingDistance": 15,
    "predatorInterception": True,
    "gridCellSize": 100,
    "fpsTarget": 60,
    "speedMultiplier": 1.0,
    "visualizeGrid": False,
    "visualizationMode": 0,
    "backgroundColor": [20, 20, 30],
    "boidColor": [200, 200, 255],
    "predatorColor": [255, 50, 50],
    "maxAgentCount": 300,
    "accelerationFactor": 1.0,
    "environmentalWind": [0.0, 0.0, 0.0],
    "adaptiveBehaviorEnabled": True,
    "learningRate": 0.05,
    "splitMergeEnabled": True,
    "scoreOutputFile": "boids_simulation_score.json"
}

# Extract commonly used values
WIDTH = CONFIG["screenWidth"]
HEIGHT = CONFIG["screenHeight"]
DEPTH = CONFIG["depth"]
FPS = CONFIG["fpsTarget"]
GRID_SIZE = CONFIG["gridCellSize"]

# Rule weights
SEPARATION_WEIGHT = 1.8
ALIGNMENT_WEIGHT = 1.2
COHESION_WEIGHT = 1.0
BOUNDARY_WEIGHT = 5.0
PREDATOR_AVOID_WEIGHT = 4.0

BOUNDARY_MARGIN = 100

# Camera Settings
CAMERA_DISTANCE = 1800
FOV = 800

# ==================== 3D VECTOR UTILS ====================
# Using pygame.math.Vector3

# ==================== SPATIAL GRID 3D ====================
class SpatialGrid3D:
    """Spatial hash grid for efficient neighbor lookup in 3D"""
    def __init__(self, width, height, depth, cell_size):
        self.width = width
        self.height = height
        self.depth = depth
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
        self.layers = int(depth / cell_size) + 1
    
    def clear(self):
        self.grid.clear()
    
    def _hash(self, x, y, z):
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        layer = int(z / self.cell_size)
        return (
            max(0, min(col, self.cols - 1)), 
            max(0, min(row, self.rows - 1)),
            max(0, min(layer, self.layers - 1))
        )
    
    def insert(self, agent):
        cell = self._hash(agent.position.x, agent.position.y, agent.position.z)
        self.grid[cell].append(agent)
    
    def get_neighbors(self, position, radius):
        """Get all agents within radius of position"""
        neighbors = []
        cell = self._hash(position.x, position.y, position.z)
        cells_to_check = self._get_adjacent_cells(cell)
        
        for c in cells_to_check:
            for agent in self.grid.get(c, []):
                dist = position.distance_to(agent.position)
                if dist < radius:
                    neighbors.append(agent)
        
        return neighbors
    
    def _get_adjacent_cells(self, cell):
        """Get cell and its 26 neighbors"""
        col, row, layer = cell
        cells = []
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                for dl in [-1, 0, 1]:
                    nc, nr, nl = col + dc, row + dr, layer + dl
                    if (0 <= nc < self.cols and 
                        0 <= nr < self.rows and 
                        0 <= nl < self.layers):
                        cells.append((nc, nr, nl))
        return cells

# ==================== CAMERA CLASS ====================
class Camera:
    def __init__(self):
        # Center of the simulation volume
        self.target = pygame.math.Vector3(WIDTH/2, HEIGHT/2, DEPTH/2)
        
        # Diagonal position: back, up, and left
        offset = pygame.math.Vector3(-1, 0.8, -1).normalize() * CAMERA_DISTANCE
        self.position = self.target + offset
        
        # Calculate initial yaw and pitch to look at target
        direction = (self.target - self.position).normalize()
        self.yaw = math.atan2(direction.z, direction.x)
        self.pitch = math.asin(direction.y)
        
        # Movement speed
        self.move_speed = 20.0
        self.rotate_speed = 0.05
        
        self._update_vectors()
        
    def _update_vectors(self):
        """Update camera basis vectors based on yaw and pitch"""
        # Calculate forward vector from yaw and pitch
        self.forward = pygame.math.Vector3(
            math.cos(self.pitch) * math.cos(self.yaw),
            math.sin(self.pitch),
            math.cos(self.pitch) * math.sin(self.yaw)
        ).normalize()
        
        # Calculate right and up vectors
        self.up_ref = pygame.math.Vector3(0, 1, 0)
        self.right = self.forward.cross(self.up_ref).normalize()
        self.up = self.right.cross(self.forward).normalize()
    
    def move(self, direction):
        """Move camera in specified direction"""
        if direction == 'forward':
            self.position += self.forward * self.move_speed
        elif direction == 'backward':
            self.position -= self.forward * self.move_speed
        elif direction == 'left':
            self.position -= self.right * self.move_speed
        elif direction == 'right':
            self.position += self.right * self.move_speed
        elif direction == 'up':
            self.position += self.up_ref * self.move_speed
        elif direction == 'down':
            self.position -= self.up_ref * self.move_speed
    
    def rotate(self, yaw_delta, pitch_delta):
        """Rotate camera by yaw and pitch deltas"""
        self.yaw += yaw_delta * self.rotate_speed
        self.pitch += pitch_delta * self.rotate_speed
        
        # Clamp pitch to avoid gimbal lock
        self.pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.pitch))
        
        self._update_vectors()
        
    def project(self, point_3d):
        """Project 3D point to 2D screen coordinates"""
        # Transform to camera space
        to_point = point_3d - self.position
        
        # Dot products to get coordinates in camera basis
        x = to_point.dot(self.right)
        y = to_point.dot(self.up)
        z = to_point.dot(self.forward)
        
        # Perspective projection
        if z <= 1: # Behind camera or too close
            return None, 0
            
        scale = FOV / z
        screen_x = int(WIDTH/2 + x * scale)
        screen_y = int(HEIGHT/2 - y * scale) # Flip Y for screen coords
        
        return (screen_x, screen_y), scale

# ==================== BASE AGENT CLASS ====================
class Agent:
    def __init__(self, x, y, z, max_speed, max_force):
        self.position = pygame.math.Vector3(x, y, z)
        
        # Random 3D velocity
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        vx = math.sin(phi) * math.cos(theta)
        vy = math.sin(phi) * math.sin(theta)
        vz = math.cos(phi)
        
        self.velocity = pygame.math.Vector3(vx, vy, vz) * random.uniform(1.0, max_speed)
        self.acceleration = pygame.math.Vector3(0, 0, 0)
        self.max_speed = max_speed * CONFIG["speedMultiplier"]
        self.max_force = max_force * CONFIG["accelerationFactor"]
    
    def apply_force(self, force):
        self.acceleration += force
    
    def update(self):
        # Apply environmental wind
        wind = pygame.math.Vector3(CONFIG["environmentalWind"])
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
    def __init__(self, x, y, z):
        super().__init__(x, y, z, CONFIG["maxSpeed"], CONFIG["maxForce"])
        self.color = CONFIG["boidColor"]
        self.adaptive_weights = {
            "separation": SEPARATION_WEIGHT,
            "alignment": ALIGNMENT_WEIGHT,
            "cohesion": COHESION_WEIGHT
        }
        self.stress_level = 0.0
    
    def flock(self, spatial_grid, predators):
        neighbors = spatial_grid.get_neighbors(self.position, CONFIG["neighborRadius"])
        neighbors = [n for n in neighbors if isinstance(n, Boid) and n is not self]
        
        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        
        boundary = self.boundary_repulsion()
        pred_avoid = self.avoid_predators(predators)
        
        if CONFIG["adaptiveBehaviorEnabled"]:
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
        steering = pygame.math.Vector3(0, 0, 0)
        total = 0
        for other in neighbors:
            dist = self.position.distance_to(other.position)
            if 0 < dist < CONFIG["desiredSeparation"]:
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
        steering = pygame.math.Vector3(0, 0, 0)
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
        steering = pygame.math.Vector3(0, 0, 0)
        total = 0
        center = pygame.math.Vector3(0, 0, 0)
        
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
        steering = pygame.math.Vector3(0, 0, 0)
        
        # X boundaries
        if self.position.x < BOUNDARY_MARGIN:
            force = pygame.math.Vector3(1, 0, 0)
            force *= (BOUNDARY_MARGIN - self.position.x) / BOUNDARY_MARGIN
            steering += force
        elif self.position.x > WIDTH - BOUNDARY_MARGIN:
            force = pygame.math.Vector3(-1, 0, 0)
            force *= (self.position.x - (WIDTH - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force
            
        # Y boundaries
        if self.position.y < BOUNDARY_MARGIN:
            force = pygame.math.Vector3(0, 1, 0)
            force *= (BOUNDARY_MARGIN - self.position.y) / BOUNDARY_MARGIN
            steering += force
        elif self.position.y > HEIGHT - BOUNDARY_MARGIN:
            force = pygame.math.Vector3(0, -1, 0)
            force *= (self.position.y - (HEIGHT - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force

        # Z boundaries
        if self.position.z < BOUNDARY_MARGIN:
            force = pygame.math.Vector3(0, 0, 1)
            force *= (BOUNDARY_MARGIN - self.position.z) / BOUNDARY_MARGIN
            steering += force
        elif self.position.z > DEPTH - BOUNDARY_MARGIN:
            force = pygame.math.Vector3(0, 0, -1)
            force *= (self.position.z - (DEPTH - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN
            steering += force
        
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def avoid_predators(self, predators):
        steering = pygame.math.Vector3(0, 0, 0)
        total = 0
        
        for pred in predators:
            dist = self.position.distance_to(pred.position)
            if dist < CONFIG["predatorAvoidRadius"]:
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
        if not CONFIG["adaptiveBehaviorEnabled"]:
            return
        
        lr = CONFIG["learningRate"]
        if self.stress_level > 0.5:
            self.adaptive_weights["separation"] += lr * 0.5
            self.adaptive_weights["cohesion"] -= lr * 0.2
        else:
            self.adaptive_weights["separation"] += lr * (SEPARATION_WEIGHT - self.adaptive_weights["separation"])
            self.adaptive_weights["alignment"] += lr * (ALIGNMENT_WEIGHT - self.adaptive_weights["alignment"])
            self.adaptive_weights["cohesion"] += lr * (COHESION_WEIGHT - self.adaptive_weights["cohesion"])
        
        for key in self.adaptive_weights:
            self.adaptive_weights[key] = max(0.1, min(3.0, self.adaptive_weights[key]))

# ==================== PREDATOR CLASS ====================
class Predator(Agent):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, CONFIG["maxSpeed"] * 0.9, CONFIG["maxForce"] * 0.9)
        self.color = CONFIG["predatorColor"]
        self.hunt_target = None
        self.catch_flash = 0
    
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
    
    def hunt(self, boids):
        caught_boid = None
        if not boids:
            return caught_boid
        
        if self.hunt_target is None or random.random() < 0.01:
            closest = min(boids, key=lambda b: self.position.distance_to(b.position))
            self.hunt_target = closest
        
        if self.hunt_target:
            distance = self.position.distance_to(self.hunt_target.position)
            
            if distance < CONFIG["catchingDistance"]:
                caught_boid = self.hunt_target
                self.hunt_target = None
                self.catch_flash = 15
                return caught_boid
            
            if CONFIG["predatorInterception"]:
                target_pos = self.calculate_interception(self.hunt_target)
            else:
                target_pos = self.hunt_target.position
            
            desired = target_pos - self.position
            if desired.length() > 0.001:
                desired.scale_to_length(self.max_speed)
                steer = self.steering(desired)
                self.apply_force(steer * 1.5)
        
        boundary = self.boundary_repulsion()
        self.apply_force(boundary)
        
        return caught_boid
    
    def boundary_repulsion(self):
        steering = pygame.math.Vector3(0, 0, 0)
        margin = BOUNDARY_MARGIN
        
        if self.position.x < margin:
            steering.x = (margin - self.position.x) / margin
        elif self.position.x > WIDTH - margin:
            steering.x = -(self.position.x - (WIDTH - margin)) / margin
            
        if self.position.y < margin:
            steering.y = (margin - self.position.y) / margin
        elif self.position.y > HEIGHT - margin:
            steering.y = -(self.position.y - (HEIGHT - margin)) / margin

        if self.position.z < margin:
            steering.z = (margin - self.position.z) / margin
        elif self.position.z > DEPTH - margin:
            steering.z = -(self.position.z - (DEPTH - margin)) / margin
        
        if steering.length() > 0.001:
            steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        
        return steering
    
    def update(self):
        super().update()
        if self.catch_flash > 0:
            self.catch_flash -= 1

# ==================== SIMULATION CLASS ====================
class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("3D Boids Simulation")
        self.clock = pygame.time.Clock()
        
        self.spatial_grid = SpatialGrid3D(WIDTH, HEIGHT, DEPTH, GRID_SIZE)
        self.camera = Camera()
        
        self.boids = []
        self.predators = []
        self._spawn_agents()
        
        self.frame_count = 0
        self.running = True
        
        self.stats = {
            "avg_speed": 0,
            "avg_stress": 0,
            "boids_eaten": 0
        }
    
    def _spawn_agents(self):
        for _ in range(CONFIG["boidCount"]):
            x = random.uniform(100, WIDTH - 100)
            y = random.uniform(100, HEIGHT - 100)
            z = random.uniform(100, DEPTH - 100)
            self.boids.append(Boid(x, y, z))
        
        for _ in range(CONFIG["predatorCount"]):
            x = random.uniform(50, WIDTH - 50)
            y = random.uniform(50, HEIGHT - 50)
            z = random.uniform(50, DEPTH - 50)
            self.predators.append(Predator(x, y, z))
    
    def update(self):
        self.spatial_grid.clear()
        for boid in self.boids:
            self.spatial_grid.insert(boid)
        
        for boid in self.boids:
            boid.flock(self.spatial_grid, self.predators)
            boid.update()
        
        caught_boids = []
        for predator in self.predators:
            caught = predator.hunt(self.boids)
            if caught:
                caught_boids.append(caught)
            predator.update()
        
        for caught in caught_boids:
            if caught in self.boids:
                self.boids.remove(caught)
                self.stats["boids_eaten"] += 1
        
        if CONFIG["splitMergeEnabled"] and self.frame_count % 100 == 0:
            self._handle_split_merge()
        
        self.frame_count += 1
        self._update_statistics()
    
    def _handle_split_merge(self):
        if len(self.boids) < CONFIG["maxAgentCount"] and random.random() < 0.1:
            x = random.uniform(100, WIDTH - 100)
            y = random.uniform(100, HEIGHT - 100)
            z = random.uniform(100, DEPTH - 100)
            self.boids.append(Boid(x, y, z))
        elif len(self.boids) > CONFIG["boidCount"] * 1.5:
            if self.boids:
                self.boids.pop(random.randint(0, len(self.boids) - 1))
    
    def _update_statistics(self):
        if not self.boids:
            return
        total_speed = sum(b.velocity.length() for b in self.boids)
        self.stats["avg_speed"] = total_speed / len(self.boids)
        total_stress = sum(b.stress_level for b in self.boids)
        self.stats["avg_stress"] = total_stress / len(self.boids)
    
    def draw(self):
        self.screen.fill(CONFIG["backgroundColor"])
        
        # Collect all agents to draw
        draw_list = []
        
        # Add boids
        for boid in self.boids:
            pos_2d, scale = self.camera.project(boid.position)
            if pos_2d:
                # Calculate distance from camera for sorting
                dist = (boid.position - self.camera.position).length_squared()
                draw_list.append({
                    'type': 'boid',
                    'agent': boid,
                    'pos': pos_2d,
                    'scale': scale,
                    'dist': dist
                })
        
        # Add predators
        for pred in self.predators:
            pos_2d, scale = self.camera.project(pred.position)
            if pos_2d:
                dist = (pred.position - self.camera.position).length_squared()
                draw_list.append({
                    'type': 'predator',
                    'agent': pred,
                    'pos': pos_2d,
                    'scale': scale,
                    'dist': dist
                })
        
        # Add boundary box corners for reference
        corners = [
            (0, 0, 0), (WIDTH, 0, 0), (WIDTH, HEIGHT, 0), (0, HEIGHT, 0),
            (0, 0, DEPTH), (WIDTH, 0, DEPTH), (WIDTH, HEIGHT, DEPTH), (0, HEIGHT, DEPTH)
        ]
        for i, corner in enumerate(corners):
            pos_2d, scale = self.camera.project(pygame.math.Vector3(corner))
            if pos_2d:
                dist = (pygame.math.Vector3(corner) - self.camera.position).length_squared()
                draw_list.append({
                    'type': 'corner',
                    'pos': pos_2d,
                    'scale': scale,
                    'dist': dist
                })
                
        # Sort by distance (painters algorithm) - farthest first
        draw_list.sort(key=lambda x: x['dist'], reverse=True)
        
        # Draw everything
        for item in draw_list:
            if item['type'] == 'corner':
                pygame.draw.circle(self.screen, (100, 100, 100), item['pos'], 2)
            elif item['type'] == 'boid':
                boid = item['agent']
                color = boid.color
                if CONFIG["visualizationMode"] == 2:
                    stress = int(255 * boid.stress_level)
                    color = (stress, boid.color[1], boid.color[2])
                
                size = max(2, int(4 * item['scale']))
                pygame.draw.circle(self.screen, color, item['pos'], size)
                
                # Draw velocity vector
                if CONFIG["visualizationMode"] >= 1:
                    end_pos_3d = boid.position + boid.velocity.normalize() * 15
                    end_pos_2d, _ = self.camera.project(end_pos_3d)
                    if end_pos_2d:
                        pygame.draw.line(self.screen, color, item['pos'], end_pos_2d, 1)
                        
            elif item['type'] == 'predator':
                pred = item['agent']
                size = max(3, int(7 * item['scale']))
                
                if pred.catch_flash > 0:
                    size = int(size * 1.5)
                    pygame.draw.circle(self.screen, (255, 255, 255), item['pos'], size)
                else:
                    pygame.draw.circle(self.screen, pred.color, item['pos'], size)
                    pygame.draw.circle(self.screen, (255, 100, 100), item['pos'], size, 2)
                
                # Debug mode: show interception point and prediction line
                if CONFIG["visualizationMode"] >= 2 and pred.hunt_target and CONFIG["predatorInterception"]:
                    intercept_pos_3d = pred.calculate_interception(pred.hunt_target)
                    intercept_pos_2d, _ = self.camera.project(intercept_pos_3d)
                    
                    if intercept_pos_2d:
                        # Draw line from predator to predicted interception point
                        pygame.draw.line(self.screen, (255, 150, 0), item['pos'], intercept_pos_2d, 1)
                        # Draw X at interception point
                        size = 5
                        pygame.draw.line(self.screen, (255, 200, 0), 
                                       (intercept_pos_2d[0] - size, intercept_pos_2d[1] - size),
                                       (intercept_pos_2d[0] + size, intercept_pos_2d[1] + size), 2)
                        pygame.draw.line(self.screen, (255, 200, 0),
                                       (intercept_pos_2d[0] + size, intercept_pos_2d[1] - size),
                                       (intercept_pos_2d[0] - size, intercept_pos_2d[1] + size), 2)

        # Draw bounding box lines (simple wireframe)
        self._draw_boundary_box()
        
        self._draw_stats()
        pygame.display.flip()
        
    def _draw_boundary_box(self):
        # Define edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Front face (z=0)
            (4, 5), (5, 6), (6, 7), (7, 4), # Back face (z=DEPTH)
            (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
        ]
        corners_3d = [
            pygame.math.Vector3(0, 0, 0), pygame.math.Vector3(WIDTH, 0, 0), 
            pygame.math.Vector3(WIDTH, HEIGHT, 0), pygame.math.Vector3(0, HEIGHT, 0),
            pygame.math.Vector3(0, 0, DEPTH), pygame.math.Vector3(WIDTH, 0, DEPTH), 
            pygame.math.Vector3(WIDTH, HEIGHT, DEPTH), pygame.math.Vector3(0, HEIGHT, DEPTH)
        ]
        
        # Project all corners
        projected = []
        for c in corners_3d:
            p, _ = self.camera.project(c)
            projected.append(p)
            
        # Draw lines
        for start, end in edges:
            p1 = projected[start]
            p2 = projected[end]
            if p1 and p2:
                pygame.draw.line(self.screen, (50, 50, 60), p1, p2, 1)

    def _draw_stats(self):
        font = pygame.font.Font(None, 24)
        y_offset = 10
        stats_text = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Boids: {len(self.boids)}",
            f"Eaten: {self.stats['boids_eaten']}",
            f"Avg Speed: {self.stats['avg_speed']:.2f}",
            f"Avg Stress: {self.stats['avg_stress']:.2f}",
            "",
            "Controls:",
            "WASD - Move camera",
            "Q/E - Up/Down",
            "Arrows - Rotate",
            "V - Viz mode"
        ]
        for text in stats_text:
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_v:
                        CONFIG["visualizationMode"] = (CONFIG["visualizationMode"] + 1) % 3
            
            # Handle camera controls
            keys = pygame.key.get_pressed()
            
            # Movement (WASD + Q/E)
            if keys[pygame.K_w]:
                self.camera.move('forward')
            if keys[pygame.K_s]:
                self.camera.move('backward')
            if keys[pygame.K_a]:
                self.camera.move('left')
            if keys[pygame.K_d]:
                self.camera.move('right')
            if keys[pygame.K_q]:
                self.camera.move('down')
            if keys[pygame.K_e]:
                self.camera.move('up')
            
            # Rotation (Arrow keys)
            yaw_delta = 0
            pitch_delta = 0
            if keys[pygame.K_LEFT]:
                yaw_delta = -1
            if keys[pygame.K_RIGHT]:
                yaw_delta = 1
            if keys[pygame.K_UP]:
                pitch_delta = 1
            if keys[pygame.K_DOWN]:
                pitch_delta = -1
            
            if yaw_delta != 0 or pitch_delta != 0:
                self.camera.rotate(yaw_delta, pitch_delta)
            
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    Simulation().run()
