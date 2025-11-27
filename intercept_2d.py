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
    "predatorCount": 1,
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
    "scoreOutputFile": "boids_simulation_score.json"
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

# ==================== PREDATOR CLASS ====================
class Predator(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, CONFIG["maxSpeed"] * 0.8, CONFIG["maxForce"] * 0.8)
        self.color = CONFIG["predatorColor"]
        self.hunt_target = None
        self.catch_flash = 0  # Visual feedback when catching prey
    
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
        # This is a simplified solution assuming constant velocities
        target_speed = target.velocity.length()
        
        # If target is stationary, aim directly at it
        if target_speed < 0.001:
            return target.position
        
        # Calculate the interception time using the formula:
        # We need to solve: |predator_pos + predator_vel*t| = |target_pos + target_vel*t|
        # Simplified approach: estimate time based on distance and relative speeds
        
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
    
    def hunt(self, boids):
        caught_boid = None
        
        if not boids:
            return caught_boid
        
        # Find closest boid
        if self.hunt_target is None or random.random() < 0.01:
            closest = min(boids, key=lambda b: self.position.distance_to(b.position))
            self.hunt_target = closest
        
        if self.hunt_target:
            # Check if close enough to catch
            distance = self.position.distance_to(self.hunt_target.position)
            
            if distance < CONFIG["catchingDistance"]:
                # Caught the boid!
                caught_boid = self.hunt_target
                self.hunt_target = None  # Clear target, will find new one
                self.catch_flash = 15  # Flash for 15 frames
                return caught_boid
            
            # Calculate target position (intercept or direct)
            if CONFIG["predatorInterception"]:
                # Predictive interception - aim where target will be
                target_pos = self.calculate_interception(self.hunt_target)
            else:
                # Simple chase - aim at current position
                target_pos = self.hunt_target.position
            
            # Continue pursuit toward target position
            desired = target_pos - self.position
            if desired.length() > 0.001:
                desired.scale_to_length(self.max_speed)
                steer = self.steering(desired)
                self.apply_force(steer * 1.5)
        
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
        """Override update to handle catch flash"""
        super().update()
        if self.catch_flash > 0:
            self.catch_flash -= 1
    
    def draw(self, surface, debug_mode=False):
        # Draw larger and brighter when catching prey
        if self.catch_flash > 0:
            radius = 6 + int(self.catch_flash * 0.6)  # Grow up to 15 pixels
            flash_color = (255, 255, 255)  # White flash
            pygame.draw.circle(surface, flash_color, (int(self.position.x), int(self.position.y)), radius)
        else:
            pygame.draw.circle(surface, self.color, (int(self.position.x), int(self.position.y)), 6)
            pygame.draw.circle(surface, (255, 100, 100), (int(self.position.x), int(self.position.y)), 6, 2)
        
        # Debug mode: show interception point and prediction line
        if debug_mode >= 2 and self.hunt_target and CONFIG["predatorInterception"]:
            intercept_pos = self.calculate_interception(self.hunt_target)
            # Draw line from predator to predicted interception point
            pygame.draw.line(surface, (255, 150, 0), self.position, intercept_pos, 1)
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
