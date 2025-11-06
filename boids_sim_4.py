import pygame
import random
import math
import collections
import numpy as np
import sys

# --- Parameters ---
WIDTH, HEIGHT = 960, 720 # Keep window size reasonable
SIM_WIDTH, SIM_HEIGHT = WIDTH, HEIGHT # Use full window now

# --- Effects Parameters (not runtime modifiable) ---
INITIAL_CLUSTER_RADIUS = 50 # How tightly boids start
TRAIL_LENGTH = 10          # Length of the trails
PREDATOR_SPAWN_TIME = 5000 # Time in milliseconds (5 seconds)
ZOOM_DURATION = 1200      # Zoom out over 12 seconds
INITIAL_ZOOM = 1.5         # Start zoomed in
FINAL_ZOOM = 1.0           # End zoom level

# Predator Settings
PREDATOR_ACTIVE = False # Controlled by timer now
PREDATOR_SPEED = 3.5
PREDATOR_CHASE_WEIGHT = 1.5

FPS = 60

# --- Parameter Dictionary for Runtime Modification ---
params = {
    # Boid parameters
    'NUM_BOIDS': 130,
    'MAX_SPEED': 2.0,
    'MAX_FORCE': 0.18,
    'PERCEPTION_RADIUS': 55,
    'SEPARATION_WEIGHT': 0.8,
    'ALIGNMENT_WEIGHT': 0.8,
    'COHESION_WEIGHT': 0.7,
    'PREDATOR_EVADE_WEIGHT': 2.5,
    
    # Shark parameters
    'PREDATOR_PERCEPTION': 180,
    'K_C': 0.2,
    'K_D': 0.1,
    'K_B': 0.3,
    'PREY_DENSITY_THRESHOLD': 25,
    'BURST_DURATION': 500,
    'CRUISE_SPEED': 1.5,
    'BURST_SPEED': 4.0,
    'MAX_ANGULAR_VELOCITY_SLOW': 0.08,
    'MAX_ANGULAR_VELOCITY_FAST': 0.03,
    'MIN_SPEED_FOR_TURNING': 0.1,
    
    # Collision
    'COLLISION_RADIUS': 30,  # Distance for shark to eat fish
}

# Function to sync parameters from dictionary to global variables
def sync_params():
    """Update global variables from params dictionary."""
    global NUM_BOIDS, MAX_SPEED, MAX_FORCE, PERCEPTION_RADIUS
    global SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT, PREDATOR_EVADE_WEIGHT
    global PREDATOR_PERCEPTION, K_C, K_D, K_B, PREY_DENSITY_THRESHOLD
    global BURST_DURATION, CRUISE_SPEED, BURST_SPEED
    global MAX_ANGULAR_VELOCITY_SLOW, MAX_ANGULAR_VELOCITY_FAST, MIN_SPEED_FOR_TURNING
    global COLLISION_RADIUS
    
    NUM_BOIDS = params['NUM_BOIDS']
    MAX_SPEED = params['MAX_SPEED']
    MAX_FORCE = params['MAX_FORCE']
    PERCEPTION_RADIUS = params['PERCEPTION_RADIUS']
    SEPARATION_WEIGHT = params['SEPARATION_WEIGHT']
    ALIGNMENT_WEIGHT = params['ALIGNMENT_WEIGHT']
    COHESION_WEIGHT = params['COHESION_WEIGHT']
    PREDATOR_EVADE_WEIGHT = params['PREDATOR_EVADE_WEIGHT']
    PREDATOR_PERCEPTION = params['PREDATOR_PERCEPTION']
    K_C = params['K_C']
    K_D = params['K_D']
    K_B = params['K_B']
    PREY_DENSITY_THRESHOLD = params['PREY_DENSITY_THRESHOLD']
    BURST_DURATION = params['BURST_DURATION']
    CRUISE_SPEED = params['CRUISE_SPEED']
    BURST_SPEED = params['BURST_SPEED']
    MAX_ANGULAR_VELOCITY_SLOW = params['MAX_ANGULAR_VELOCITY_SLOW']
    MAX_ANGULAR_VELOCITY_FAST = params['MAX_ANGULAR_VELOCITY_FAST']
    MIN_SPEED_FOR_TURNING = params['MIN_SPEED_FOR_TURNING']
    COLLISION_RADIUS = params['COLLISION_RADIUS']

# Initialize parameters
sync_params()

# --- Pygame Setup ---
pygame.init()
# Try hardware acceleration flags
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
# screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Fallback if flags cause issues
pygame.display.set_caption("Boids Zoom Out Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 18)

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# Trail color sequence (fades to black)
TRAIL_COLORS = [pygame.Color(c, c, int(c*1.2)) for c in range(180, 50, -13)] # Fading Blue/White
# Ensure trail colors list matches TRAIL_LENGTH
if len(TRAIL_COLORS) < TRAIL_LENGTH:
     TRAIL_COLORS += [TRAIL_COLORS[-1]] * (TRAIL_LENGTH - len(TRAIL_COLORS))
elif len(TRAIL_COLORS) > TRAIL_LENGTH:
     TRAIL_COLORS = TRAIL_COLORS[:TRAIL_LENGTH]

BOID_COLOR = TRAIL_COLORS[0] # Make boid the brightest part of trail
PREDATOR_COLOR = (255, 50, 50)

# --- Camera State ---
current_zoom = INITIAL_ZOOM
camera_center_world = pygame.Vector2(SIM_WIDTH / 2, SIM_HEIGHT / 2) # Point the world focuses on

# --- Transformation Function ---
def world_to_screen(world_pos):
    """Converts world coordinates to screen coordinates based on zoom."""
    # Vector from focus point in world space
    relative_pos = world_pos - camera_center_world
    # Scale by zoom and add screen center offset
    screen_pos = relative_pos * current_zoom + pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    return screen_pos

# --- Settings Panel Class ---
class SettingsPanel:
    def __init__(self):
        self.visible = False
        self.selected_index = 0
        self.editing = False
        self.edit_buffer = ""
        self.param_keys = list(params.keys())
        self.scroll_offset = 0
        self.max_visible = 15
        
    def toggle(self):
        self.visible = not self.visible
        if not self.visible:
            self.editing = False
            self.edit_buffer = ""
    
    def handle_event(self, event):
        if not self.visible:
            return False
        
        if event.type == pygame.KEYDOWN:
            if self.editing:
                if event.key == pygame.K_RETURN:
                    # Apply the edited value
                    try:
                        value = float(self.edit_buffer)
                        key = self.param_keys[self.selected_index]
                        if key == 'NUM_BOIDS':
                            value = int(value)
                            value = max(1, min(500, value))  # Limit boid count
                        params[key] = value
                        sync_params()
                        self.editing = False
                        self.edit_buffer = ""
                        return True
                    except ValueError:
                        self.editing = False
                        self.edit_buffer = ""
                elif event.key == pygame.K_ESCAPE:
                    self.editing = False
                    self.edit_buffer = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.edit_buffer = self.edit_buffer[:-1]
                elif event.unicode.isdigit() or event.unicode == '.' or event.unicode == '-':
                    self.edit_buffer += event.unicode
            else:
                if event.key == pygame.K_UP:
                    self.selected_index = max(0, self.selected_index - 1)
                    if self.selected_index < self.scroll_offset:
                        self.scroll_offset = self.selected_index
                elif event.key == pygame.K_DOWN:
                    self.selected_index = min(len(self.param_keys) - 1, self.selected_index + 1)
                    if self.selected_index >= self.scroll_offset + self.max_visible:
                        self.scroll_offset = self.selected_index - self.max_visible + 1
                elif event.key == pygame.K_RETURN:
                    self.editing = True
                    self.edit_buffer = str(params[self.param_keys[self.selected_index]])
                elif event.key == pygame.K_ESCAPE:
                    self.toggle()
        return False
    
    def draw(self, surface):
        if not self.visible:
            return
        
        # Draw semi-transparent background
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        surface.blit(overlay, (0, 0))
        
        # Panel dimensions
        panel_width = 500
        panel_height = min(600, len(self.param_keys) * 25 + 60)
        panel_x = (WIDTH - panel_width) // 2
        panel_y = (HEIGHT - panel_height) // 2
        
        # Draw panel background
        pygame.draw.rect(surface, (40, 40, 40), (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(surface, WHITE, (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = font.render("Settings (Press ESC to close, Enter to edit)", True, WHITE)
        surface.blit(title, (panel_x + 10, panel_y + 10))
        
        # Draw parameters
        y_offset = panel_y + 50
        visible_keys = self.param_keys[self.scroll_offset:self.scroll_offset + self.max_visible]
        
        for i, key in enumerate(visible_keys):
            actual_index = self.scroll_offset + i
            y_pos = y_offset + i * 25
            
            # Highlight selected
            if actual_index == self.selected_index:
                pygame.draw.rect(surface, (60, 60, 100), (panel_x + 5, y_pos - 2, panel_width - 10, 23))
            
            # Parameter name
            name_text = small_font.render(key + ":", True, WHITE)
            surface.blit(name_text, (panel_x + 15, y_pos))
            
            # Parameter value
            if self.editing and actual_index == self.selected_index:
                value_text = small_font.render(self.edit_buffer + "_", True, (100, 255, 100))
            else:
                value_text = small_font.render(str(params[key]), True, (200, 200, 200))
            surface.blit(value_text, (panel_x + 250, y_pos))
        
        # Instructions
        if not self.editing:
            inst_text = small_font.render("Use Up/Down to navigate, Enter to edit, ESC to close", True, (150, 150, 150))
            surface.blit(inst_text, (panel_x + 10, panel_y + panel_height - 25))

# --- Boid Class (with Trails and Zoom-aware Drawing) ---
class Boid:
    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y) # World position
        angle = random.uniform(0, 2 * math.pi)
        # Start with slightly lower initial speed from center
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, MAX_SPEED * 0.8)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE
        self.perception = PERCEPTION_RADIUS
        self.color = BOID_COLOR
        self.trail = collections.deque(maxlen=TRAIL_LENGTH)

    def apply_force(self, force):
        self.acceleration += force

    def wrap_edges(self):
        # Wrap edges in world coordinates
        if self.position.x > SIM_WIDTH: self.position.x = 0
        elif self.position.x < 0: self.position.x = SIM_WIDTH
        if self.position.y > SIM_HEIGHT: self.position.y = 0
        elif self.position.y < 0: self.position.y = SIM_HEIGHT

    def update(self):
        # Store current position for trail *before* updating
        self.trail.append(self.position.copy())

        self.velocity += self.acceleration
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        self.position += self.velocity
        self.acceleration *= 0

    def steer(self, desired_velocity):
        steer = desired_velocity - self.velocity
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    # --- Separation, Alignment, Cohesion (Logic Unchanged) ---
    def separation(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < self.perception / 1.5:
                diff = self.position - other.position
                if diff.length_squared() > 0:
                     diff /= distance * distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            if steering.length() > 0: steering.scale_to_length(self.max_speed)
            steering = self.steer(steering)
        return steering

    def alignment(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < self.perception:
                steering += other.velocity
                total += 1
        if total > 0:
            steering /= total
            if steering.length() > 0: steering.scale_to_length(self.max_speed)
            steering = self.steer(steering)
        return steering

    def cohesion(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        center_of_mass = pygame.Vector2(0, 0)
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < self.perception:
                center_of_mass += other.position
                total += 1
        if total > 0:
            center_of_mass /= total
            desired_velocity = center_of_mass - self.position
            if desired_velocity.length() > 0: desired_velocity.scale_to_length(self.max_speed)
            steering = self.steer(desired_velocity)
        return steering

    def evade(self, predator_pos):
        steering = pygame.Vector2(0, 0)
        if predator_pos:
            distance = self.position.distance_to(predator_pos)
            evade_radius = self.perception * 1.8 # React strongly further away
            if 0 < distance < evade_radius:
                desired_velocity = self.position - predator_pos
                # Scale force more strongly when closer
                desired_velocity = desired_velocity.normalize() * self.max_speed * (evade_radius / max(distance, 1)) # Avoid div by zero
                steering = self.steer(desired_velocity)
                # Allow evasion to be stronger than normal steering
                if steering.length() > self.max_force * 1.8:
                     steering.scale_to_length(self.max_force * 1.8)
        return steering

    def update_params(self):
        """Update parameters from global params dictionary."""
        self.max_speed = params['MAX_SPEED']
        self.max_force = params['MAX_FORCE']
        self.perception = params['PERCEPTION_RADIUS']
    
    def flock(self, boids, predator_pos=None):
        # Update parameters dynamically
        self.update_params()
        
        # Optimization: Only check neighbors within perception * squared distance
        perception_sq = self.perception * self.perception
        neighbors = [other for other in boids if other is not self and self.position.distance_squared_to(other.position) < perception_sq]

        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        evd = self.evade(predator_pos)

        self.apply_force(sep * params['SEPARATION_WEIGHT'])
        self.apply_force(ali * params['ALIGNMENT_WEIGHT'])
        self.apply_force(coh * params['COHESION_WEIGHT'])
        self.apply_force(evd * params['PREDATOR_EVADE_WEIGHT'])

    def draw(self, surface):
        # --- Draw Trail ---
        # Draw older points first (smaller/dimmer)
        for i, pos in enumerate(reversed(self.trail)):
             if i >= len(TRAIL_COLORS): break # Safety check
             screen_pos = world_to_screen(pos)
             # Scale trail point size by zoom
             size = max(1, (TRAIL_LENGTH - i) / (TRAIL_LENGTH * 1.5) * (6 / current_zoom))
             # Draw simple circles for trails (faster than lines)
             pygame.draw.circle(surface, TRAIL_COLORS[i], screen_pos, size)


        # --- Draw Boid (Triangle Head) ---
        screen_pos = world_to_screen(self.position)
        angle = math.atan2(self.velocity.y, self.velocity.x)
        # Scale boid size by zoom
        base_size = 7
        size = max(2, base_size / current_zoom)

        p1 = screen_pos + pygame.Vector2(size, 0).rotate_rad(angle)
        p2 = screen_pos + pygame.Vector2(-size / 2, size / 2).rotate_rad(angle)
        p3 = screen_pos + pygame.Vector2(-size / 2, -size / 2).rotate_rad(angle)

        pygame.draw.polygon(surface, self.color, [p1, p2, p3])


# --- Predator Class (Modified Drawing) ---
class Predator(Boid):
     def __init__(self, x, y):
          super().__init__(x, y)
          self.color = PREDATOR_COLOR
          self.max_speed = params['CRUISE_SPEED']  # Start in cruising mode
          self.perception = params['PREDATOR_PERCEPTION']
          self.max_force = params['MAX_FORCE'] * 1.2
          self.trail = collections.deque(maxlen=TRAIL_LENGTH // 2) # Shorter trail for predator
          
          # Shark behavior state
          self.state = "herding"  # "herding" or "burst"
          self.burst_start_time = None  # Track when burst mode started
          
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
    
     def update_params(self):
        """Update parameters from global params dictionary."""
        super().update_params()
        self.perception = params['PREDATOR_PERCEPTION']
        if self.state == "herding":
            self.max_speed = params['CRUISE_SPEED']
        elif self.state == "burst":
            self.max_speed = params['BURST_SPEED']
    
     def herding_acceleration(self, prey_centroid):
        """Herding acceleration: a_s = k_c(x_prey_centroid - x_s) - k_d*v_s"""
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        # Centering force toward prey centroid
        centering_force = prey_centroid - self.position
        centering_force = centering_force * params['K_C']
        
        # Damping force (opposes velocity)
        damping_force = -self.velocity * params['K_D']
        
        # Total acceleration
        acceleration = centering_force + damping_force
        return acceleration
    
     def burst_acceleration(self, prey_centroid):
        """Burst acceleration: a_s = k_b(x_prey_centroid - x_s)"""
        if prey_centroid is None:
            return pygame.Vector2(0, 0)
        
        # Direct acceleration toward prey centroid
        direction = prey_centroid - self.position
        acceleration = direction * params['K_B']
        return acceleration
    
     def constrain_turning(self, desired_velocity):
        """Constrain velocity direction change based on current speed for realistic movement."""
        current_speed = self.velocity.length()
        desired_speed = desired_velocity.length()
        
        # If desired velocity is zero, return as is
        if desired_speed < 0.001:
            return desired_velocity
        
        # If moving very slowly, allow free turning
        if current_speed < params['MIN_SPEED_FOR_TURNING']:
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
        max_angular_velocity = params['MAX_ANGULAR_VELOCITY_SLOW'] * (1 - speed_ratio) + params['MAX_ANGULAR_VELOCITY_FAST'] * speed_ratio
        
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
            if prey_density >= params['PREY_DENSITY_THRESHOLD']:
                self.state = "burst"
                self.max_speed = params['BURST_SPEED']
                self.burst_start_time = current_time
        elif self.state == "burst":
            # Transition back to herding after burst duration
            if self.burst_start_time and (current_time - self.burst_start_time) >= params['BURST_DURATION']:
                self.state = "herding"
                self.max_speed = params['CRUISE_SPEED']
                self.burst_start_time = None
    
     def update(self):
        """Override update to apply realistic turning constraints."""
        # Store current position for trail *before* updating
        self.trail.append(self.position.copy())
        
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

     def flock(self, boids, predator_pos=None):
        """Shark behavior: herding or burst based on state."""
        # Update parameters dynamically
        self.update_params()
        
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

     def draw(self, surface): # Draw predator differently
         # --- Predator Trail (Optional, shorter/different color) ---
         # Similar trail drawing logic as Boid, maybe use red shades

         # --- Draw Predator (Shark Shape) ---
        screen_pos = world_to_screen(self.position)
        angle = math.atan2(self.velocity.y, self.velocity.x)
        base_size = 18  # Increased size for more prominent shark
        size = max(4, base_size / current_zoom) # Make predator bigger & scale
        
        # Change color based on state: brighter red during burst
        if self.state == "burst":
            draw_color = (255, 100, 100)  # Brighter red during burst
        else:
            draw_color = self.color  # Normal red during herding

        # Create shark-like shape with multiple points
        # Points are relative to center, then rotated by angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        def rotate_point(x, y):
            """Rotate a point around origin by angle."""
            return pygame.Vector2(
                x * cos_a - y * sin_a,
                x * sin_a + y * cos_a
            )
        
        # Shark body points (pointing right, then rotated)
        # Nose (front point)
        nose = rotate_point(size * 1.2, 0)
        
        # Top of head
        head_top = rotate_point(size * 0.6, -size * 0.3)
        
        # Dorsal fin (top fin)
        dorsal_fin = rotate_point(size * 0.1, -size * 0.8)
        
        # Back top
        back_top = rotate_point(-size * 0.3, -size * 0.4)
        
        # Tail top
        tail_top = rotate_point(-size * 1.1, -size * 0.2)
        
        # Tail tip
        tail_tip = rotate_point(-size * 1.4, 0)
        
        # Tail bottom
        tail_bottom = rotate_point(-size * 1.1, size * 0.2)
        
        # Back bottom
        back_bottom = rotate_point(-size * 0.3, size * 0.4)
        
        # Pectoral fin (side fin) - bottom
        pectoral = rotate_point(size * 0.2, size * 0.5)
        
        # Bottom of head
        head_bottom = rotate_point(size * 0.6, size * 0.3)
        
        # Create shark polygon points
        shark_points = [
            screen_pos + nose,           # Nose
            screen_pos + head_top,       # Top of head
            screen_pos + dorsal_fin,     # Dorsal fin
            screen_pos + back_top,       # Back top
            screen_pos + tail_top,       # Tail top
            screen_pos + tail_tip,       # Tail tip
            screen_pos + tail_bottom,    # Tail bottom
            screen_pos + back_bottom,    # Back bottom
            screen_pos + pectoral,       # Pectoral fin
            screen_pos + head_bottom,    # Bottom of head
        ]
        
        # Draw filled shark body
        pygame.draw.polygon(surface, draw_color, shark_points)
        
        # Draw outline for better definition
        pygame.draw.polygon(surface, (200, 0, 0), shark_points, 1)
        
        # Draw eye
        eye_pos = screen_pos + rotate_point(size * 0.8, -size * 0.15)
        pygame.draw.circle(surface, (255, 255, 255), eye_pos, max(2, size / 8))
        pygame.draw.circle(surface, (0, 0, 0), eye_pos, max(1, size / 12))


# --- Initialization ---
# Start boids in a cluster around the world center
flock = []
for _ in range(NUM_BOIDS):
    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(0, INITIAL_CLUSTER_RADIUS)
    start_x = camera_center_world.x + math.cos(angle) * radius
    start_y = camera_center_world.y + math.sin(angle) * radius
    flock.append(Boid(start_x, start_y))

predator = None
start_time = pygame.time.get_ticks() # Time tracking for events
settings_panel = SettingsPanel()
fish_eaten = 0  # Counter for eaten fish

# --- Main Loop ---
running = True
paused = False
while running:
    current_time = pygame.time.get_ticks()
    elapsed_time = current_time - start_time

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if settings_panel.visible:
                    settings_panel.toggle()
                else:
                    running = False
            if event.key == pygame.K_SPACE: # Manual pause still works
                paused = not paused
            if event.key == pygame.K_TAB:  # Toggle settings panel
                settings_panel.toggle()
        
        # Handle settings panel events
        if settings_panel.handle_event(event):
            # If a parameter was changed, we might need to update existing boids
            # For now, parameters update dynamically in flock() methods
            pass

    if paused:
        # Draw paused text if needed
        pygame.display.flip()
        clock.tick(10)
        continue

    # --- Timed Events ---
    # Predator Spawn
    if not PREDATOR_ACTIVE and elapsed_time >= PREDATOR_SPAWN_TIME:
        if predator is None:
             pred_x = random.uniform(camera_center_world.x - 100, camera_center_world.x + 100)
             pred_y = random.uniform(camera_center_world.y - 100, camera_center_world.y + 100)
             predator = Predator(pred_x, pred_y)
             print("Predator Spawned!")
        PREDATOR_ACTIVE = True

    # Camera Zoom
    if elapsed_time < ZOOM_DURATION:
        progress = elapsed_time / ZOOM_DURATION
        # Linear interpolation for zoom
        current_zoom = INITIAL_ZOOM + (FINAL_ZOOM - INITIAL_ZOOM) * progress
        # Ease-out effect (optional): progress = 1 - (1 - progress)**2 # Quadratic ease-out
        # current_zoom = INITIAL_ZOOM + (FINAL_ZOOM - INITIAL_ZOOM) * progress
    else:
        current_zoom = FINAL_ZOOM


    # --- Update Boids & Predator ---
    predator_pos = predator.position if predator else None
    # Update all boids
    for boid in flock:
        boid.flock(flock, predator_pos)
        boid.update()
        boid.wrap_edges()
    # Update predator
    if predator:
        predator.flock(flock)
        predator.update()
        predator.wrap_edges()
        
        # Check for collisions (shark eating fish)
        collision_radius_sq = params['COLLISION_RADIUS'] * params['COLLISION_RADIUS']
        for boid in list(flock):  # Use list() to create a copy for safe iteration
            dist_sq = predator.position.distance_squared_to(boid.position)
            if dist_sq < collision_radius_sq:
                flock.remove(boid)
                fish_eaten += 1
                # Optionally add new boid if NUM_BOIDS increased
                if len(flock) < params['NUM_BOIDS']:
                    # Add new boid at random position
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(0, INITIAL_CLUSTER_RADIUS)
                    start_x = camera_center_world.x + math.cos(angle) * radius
                    start_y = camera_center_world.y + math.sin(angle) * radius
                    flock.append(Boid(start_x, start_y))
    
    # Remove excess boids if NUM_BOIDS decreased
    while len(flock) > params['NUM_BOIDS']:
        flock.pop()


    # --- Drawing ---
    screen.fill(BLACK) # Clear screen

    # Draw all boids (and their trails)
    for boid in flock:
        boid.draw(screen)
    # Draw predator
    if predator:
        predator.draw(screen)

    # --- Display Fish Eaten Counter ---
    counter_text = font.render(f"Fish Eaten: {fish_eaten}", True, WHITE)
    screen.blit(counter_text, (10, 10))
    
    # Display remaining fish count
    remaining_text = small_font.render(f"Remaining: {len(flock)}", True, (200, 200, 200))
    screen.blit(remaining_text, (10, 40))
    
    # Display settings hint
    if not settings_panel.visible:
        hint_text = small_font.render("Press TAB for Settings", True, (150, 150, 150))
        screen.blit(hint_text, (WIDTH - 180, 10))

    # --- Draw Settings Panel ---
    settings_panel.draw(screen)

    # --- Update Display ---
    pygame.display.flip()

    # --- Tick Clock ---
    clock.tick(FPS)

pygame.quit()
sys.exit()