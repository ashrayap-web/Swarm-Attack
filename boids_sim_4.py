import pygame
import random
import math
import collections
import numpy as np
import sys

# --- Parameters ---
WIDTH, HEIGHT = 960, 720 # Keep window size reasonable
SIM_WIDTH, SIM_HEIGHT = WIDTH, HEIGHT # Use full window now

NUM_BOIDS = 350      # More boids for visual impact
MAX_SPEED = 2.5      # Slightly faster movement
MAX_FORCE = 0.18     # Steering force limit
PERCEPTION_RADIUS = 55 # Perception radius

# Rule Weights (Adjusted for visual flow)
SEPARATION_WEIGHT = 0.8
ALIGNMENT_WEIGHT = 0.8 # Slightly less rigid alignment
COHESION_WEIGHT = 0.7  # Slightly less clumping

# --- Effects Parameters ---
INITIAL_CLUSTER_RADIUS = 50 # How tightly boids start
TRAIL_LENGTH = 10          # Length of the trails
PREDATOR_SPAWN_TIME = 5000 # Time in milliseconds (5 seconds)
ZOOM_DURATION = 12000      # Zoom out over 12 seconds
INITIAL_ZOOM = 3.0         # Start zoomed in
FINAL_ZOOM = 1.0           # End zoom level

# Predator Settings
PREDATOR_ACTIVE = False # Controlled by timer now
PREDATOR_SPEED = 3.5
PREDATOR_PERCEPTION = 180
PREDATOR_CHASE_WEIGHT = 1.5
PREDATOR_EVADE_WEIGHT = 2.5 # Boids react strongly

FPS = 60

# --- Pygame Setup ---
pygame.init()
# Try hardware acceleration flags
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
# screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Fallback if flags cause issues
pygame.display.set_caption("Boids Zoom Out Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

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

    def flock(self, boids, predator_pos=None):
        # Optimization: Only check neighbors within perception * squared distance
        perception_sq = self.perception * self.perception
        neighbors = [other for other in boids if other is not self and self.position.distance_squared_to(other.position) < perception_sq]

        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        evd = self.evade(predator_pos)

        self.apply_force(sep * SEPARATION_WEIGHT)
        self.apply_force(ali * ALIGNMENT_WEIGHT)
        self.apply_force(coh * COHESION_WEIGHT)
        self.apply_force(evd * PREDATOR_EVADE_WEIGHT)

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
          self.max_speed = PREDATOR_SPEED
          self.perception = PREDATOR_PERCEPTION
          self.max_force = MAX_FORCE * 1.2
          self.trail = collections.deque(maxlen=TRAIL_LENGTH // 2) # Shorter trail for predator

     def chase(self, boids):
        steering = pygame.Vector2(0, 0)
        nearest_boid_pos = None
        min_dist_sq = float('inf')
        center_of_mass = pygame.Vector2(0, 0)
        total = 0
        perception_sq = self.perception * self.perception

        for boid in boids:
            dist_sq = self.position.distance_squared_to(boid.position)
            if dist_sq < perception_sq:
                center_of_mass += boid.position
                total += 1
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    nearest_boid_pos = boid.position

        if total > 0:
             target = center_of_mass / total # Chase center of visible mass
             desired_velocity = target - self.position
             if desired_velocity.length() > 0: desired_velocity.scale_to_length(self.max_speed)
             steering = self.steer(desired_velocity)
        return steering

     def flock(self, boids, predator_pos=None): # Predator chases
          chase_force = self.chase(boids)
          self.apply_force(chase_force * PREDATOR_CHASE_WEIGHT)

     def draw(self, surface): # Draw predator differently
         # --- Predator Trail (Optional, shorter/different color) ---
         # Similar trail drawing logic as Boid, maybe use red shades

         # --- Draw Predator ---
        screen_pos = world_to_screen(self.position)
        angle = math.atan2(self.velocity.y, self.velocity.x)
        base_size = 11
        size = max(3, base_size / current_zoom) # Make predator bigger & scale

        p1 = screen_pos + pygame.Vector2(size, 0).rotate_rad(angle)
        p2 = screen_pos + pygame.Vector2(-size/1.5, size/1.5).rotate_rad(angle)
        p3 = screen_pos + pygame.Vector2(-size/1.5, -size/1.5).rotate_rad(angle)

        pygame.draw.polygon(surface, self.color, [p1, p2, p3])


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
                running = False
            if event.key == pygame.K_SPACE: # Manual pause still works
                paused = not paused

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


    # --- Drawing ---
    screen.fill(BLACK) # Clear screen

    # Draw all boids (and their trails)
    for boid in flock:
        boid.draw(screen)
    # Draw predator
    if predator:
        predator.draw(screen)

    # --- Display Info (Optional, keep minimal) ---
    # fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, WHITE)
    # screen.blit(fps_text, (5, 5))


    # --- Update Display ---
    pygame.display.flip()

    # --- Tick Clock ---
    clock.tick(FPS)

pygame.quit()
sys.exit()