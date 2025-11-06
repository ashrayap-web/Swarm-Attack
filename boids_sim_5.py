import pygame
import random
import math
import sys

# Basic Parameters for Pygame
WIDTH, HEIGHT = 960, 720
FPS = 60

# Parameters for the Boids Algorithm
NUMBER_OF_BOIDS = 350
MAX_SPEED = 3.0             # max speed of each boid
MAX_FORCE = 0.18             # max steering force (limit of acceleration)
PERCEPTION_RADIUS = 55      # this measures how far the boids can see its neighbours
SEPARATION_RADIUS = 15      # this is the measure of the radius for separation between the boids

# Rule Weights - we start with separation being the strongest
SEPARATION_WEIGHT = 1.5     
ALIGNMENT_WEIGHT = 0.8
COHESION_WEIGHT = 0.9

# Setting up Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Simulation")
clock = pygame.time.Clock()

# Colors - Pygame
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)

# Class definitions for the Boids
class Boid:
    def __init__(self, x, y):
        self.position = pygame.Vector2(x,y)
        angle = random.uniform(0, 2*math.pi) # set initial direction to random
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1.0, MAX_SPEED) # Scale speed and ensure it doesn't exceed MAX_SPEED
        self.acceleration = pygame.Vector2(0,0) # initial acceleration is 0

        # Parameters - store from the global parameters
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE
        self.perception = PERCEPTION_RADIUS
        self.separation_radius = SEPARATION_RADIUS

    # Method to add force vector to the acceleration of the boid
    def apply_force(self, force):
        self.acceleration += force

    # Method to wrap the boid around the screen edges
    def edge_wrap(self):
        if self.position.x > WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = WIDTH
        
        if self.position.y > HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = HEIGHT

    # Update the boid's velocity and position using Euler integration
    def update(self):
        self.velocity += self.acceleration
        # Set Speed Limits
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        self.position += self.velocity

        # Reset acceleration for the next frame
        self.acceleration *= 0

    # Draw the boid - Circle
    def draw(self, surface):
        pygame.draw.circle(surface, (0,255,0), self.position, 3)

    # Helper method - to calculate the steering force towards the desired velocity
    def steering(self, des_vel):
        steer = des_vel - self.velocity
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    # ---- Methods for Boids Rules ----
    # Calculate the separation steering force
    def separation(self, boids):
        steering = pygame.Vector2(0,0)
        total = 0
        # Check if other boids are within the separation radius and steer away
        for i in boids:
            try:
                dist = self.position.distance_to(i.position)
                if 0 < dist < self.separation_radius:
                    diff = self.position - i.position
                    if dist > 0:
                        diff /= dist**2
                    steering += diff
                    total += 1
            except OverflowError:
                print(f"Overflow error occurs when calculating the distance between {self.position} and {i.position}")
                continue

        # Average the steering vectors and calculate the new steering force
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        return steering

    # Calculate the alignment steering force    
    def alignment(self, boids):
        steering = pygame.Vector2(0,0)
        total = 0
        # Check if other boids are within the perception radius and steer together
        for i in boids:
            try:
                dist = self.position.distance_to(i.position)
                if 0 < dist < self.perception:
                    steering += i.velocity
                    total += 1
            except OverflowError:
                print(f"Overflow error occurs when calculating the distance between {self.position} and {i.position}")
                continue

        # Average the steering vectors and calculate the new steering force
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering.scale_to_length(self.max_speed)
            steering = self.steering(steering)
        return steering

    def cohesion(self, boids):
        steering = pygame.Vector2(0,0)
        total = 0
        com = pygame.Vector2(0,0) # Center of Mass
        # Check if other boids are within the perception radius and steer together
        for i in boids:
            try:
                dist = self.position.distance_to(i.position)
                if 0 < dist < self.perception:
                    com += i.position # sum positions
                    total += 1
            except OverflowError:
                print(f"Overflow error occurs when calculating the distance between {self.position} and {i.position}")
                continue

        # Average the steering vectors and calculate the new steering force
        if total > 0:
            com /= total
            des_pos = com - self.position
            steering /= total
            if des_pos.length() > 0:
                des_pos.scale_to_length(self.max_speed)
            steering = self.steering(des_pos)
        return steering

    # Method to calculate and apply all steering forces
    def flock(self, boids):
        neighbors = [i for i in boids if i is not self] # Find all the neighbors

        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)

        # Apply weighted forces
        self.apply_force(sep * SEPARATION_WEIGHT)
        self.apply_force(ali * ALIGNMENT_WEIGHT)
        self.apply_force(coh * COHESION_WEIGHT)

# Initializing the Flock
flock = []
for _ in range(NUMBER_OF_BOIDS):
    start_x = random.uniform(0, WIDTH)
    start_y = random.uniform(0, HEIGHT)
    flock.append(Boid(start_x, start_y))

# Main Loop
running = True
while running:
    # Event Handling - Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Update the Simulation - Boids
    for b in flock:
        b.edge_wrap()      # Handle boundaries
        b.flock(flock)      # Calculate the forces based on neighbours
        b.update()          # Apply the forces and move accordingly

    # Drawing - Pygame
    screen.fill(BLACK)
    
    # Drawing all the boids
    for b in flock:
        b.draw(screen)

    # Update the display - Pygame
    pygame.display.flip()

    # Clock Ticking - Pygame
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()
