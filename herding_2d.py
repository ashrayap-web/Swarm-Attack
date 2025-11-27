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
    "predatorCount": 3,
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
    "preyDensityThreshold": 25,
    "burstDuration": 500,
    "cruiseSpeed": 1.5,
    "burstSpeed": 4.0,
    "maxAngularVelocitySlow": 0.08,
    "maxAngularVelocityFast": 0.03,
    "minSpeedForTurning": 0.1,
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
    
    def burst_acceleration(self, prey_centroid):
        """Burst acceleration: a_s = k_b(x_prey_centroid - x_s)"""
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        # Direct acceleration toward prey centroid
        direction = prey_centroid - self.position
        acceleration = direction * CONFIG['K_B']
        return acceleration
    
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
    
    def hunt(self, boids):
        """Shark behavior: herding or burst based on state."""
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
            # Burst mode: a_s = k_b(x_prey_centroid - x_s)
            if prey_centroid is not None:
                acceleration = self.burst_acceleration(prey_centroid)
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
    
    def draw(self, surface, debug_mode=False):
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
        
        # Debug mode: show perception radius and prey centroid
        if debug_mode >= 2:
            pygame.draw.circle(surface, (100, 50, 50), (int(self.position.x), int(self.position.y)), 
                             int(self.perception), 1)

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
        
        self.frame_count = 0
        self.running = True
        
        # Statistics
        self.stats = {
            "avg_speed": 0,
            "avg_neighbors": 0,
            "avg_stress": 0,
            "flock_cohesion": 0,
            "boids_eaten": 0
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
        
        # Update predators and check for caught boids
        caught_boids = []
        for predator in self.predators:
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
    
    def draw(self):
        self.screen.fill(CONFIG["backgroundColor"])
        
        # Draw grid if enabled
        if CONFIG["visualizeGrid"] or CONFIG["visualizationMode"] == 1:
            self._draw_grid()
        
        # Draw boids
        debug_mode = CONFIG["visualizationMode"]
        for boid in self.boids:
            boid.draw(self.screen, debug_mode)
        
        # Draw predators
        for predator in self.predators:
            predator.draw(self.screen, debug_mode)
        
        # Draw stats
        self._draw_stats()
        
        pygame.display.flip()
    
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
    print("Advanced Boids Simulation")
    print("Controls:")
    print("  ESC - Quit")
    print("  G - Toggle grid visualization")
    print("  V - Cycle visualization modes")
    print("  SPACE - Save score to JSON")
    print("\nStarting simulation...")
    
    sim = Simulation()
    sim.run()
