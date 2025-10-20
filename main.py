
import pygame
import random
import math

# Simulation parameters
WIDTH, HEIGHT = 1000, 800
FPS = 60
FISH_COUNT = 60
FISH_SIZE = 5
SHARK_SIZE = 15
FISH_SPEED = 7
SHARK_SPEED = 13
FISH_PERCEPTION_RADIUS = 100
FISH_AVOIDANCE_RADIUS = 15
SHARK_ATTACK_RADIUS = 10
NOISE_LEVEL = 0.1
TIME_MULTIPLIER = 8

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def limit(self, max_val):
        if self.magnitude() > max_val:
            return self.normalize() * max_val
        return self

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Fish:
    def __init__(self, x, y):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * FISH_SPEED
        self.acceleration = Vector2D(0, 0)
        self.size = FISH_SIZE
        self.color = BLUE
        self.max_speed = FISH_SPEED
        self.max_force = 0.9 # How quickly fish can change direction
        self.perception_radius = FISH_PERCEPTION_RADIUS
        self.avoidance_radius = FISH_AVOIDANCE_RADIUS

    def apply_force(self, force):
        self.acceleration += force

    def update(self, fish_list, shark_position, dt):
        # Apply Boids rules
        alignment = self.align(fish_list)
        cohesion = self.cohere(fish_list)
        separation = self.separate(fish_list)
        shark_avoidance = self.avoid_shark(shark_position)
        wall_avoidance = self.avoid_walls()

        # Add noise to perceived forces
        alignment += Vector2D(random.uniform(-NOISE_LEVEL, NOISE_LEVEL), random.uniform(-NOISE_LEVEL, NOISE_LEVEL))
        cohesion += Vector2D(random.uniform(-NOISE_LEVEL, NOISE_LEVEL), random.uniform(-NOISE_LEVEL, NOISE_LEVEL))
        separation += Vector2D(random.uniform(-NOISE_LEVEL, NOISE_LEVEL), random.uniform(-NOISE_LEVEL, NOISE_LEVEL))
        shark_avoidance += Vector2D(random.uniform(-NOISE_LEVEL, NOISE_LEVEL), random.uniform(-NOISE_LEVEL, NOISE_LEVEL))

        # Weight forces
        if shark_avoidance.magnitude() > 0:
            # If shark is close, prioritize fleeing
            self.apply_force(alignment * 0.2)
            self.apply_force(cohesion * 0.2)
            self.apply_force(separation * 1.0)
            self.apply_force(shark_avoidance * 5.0)
        else:
            self.apply_force(alignment * 1.0)
            self.apply_force(cohesion * 1.0)
            self.apply_force(separation * 1.5)
        
        self.apply_force(wall_avoidance * 1.0) # Strong wall avoidance

        # Update velocity and position using ODEs (Euler integration)
        self.velocity += self.acceleration * dt
        self.velocity = self.velocity.limit(self.max_speed)
        self.position += self.velocity * dt
        self.acceleration *= 0 # Reset acceleration

    def align(self, fish_list):
        steering = Vector2D(0, 0)
        total = 0
        for other in fish_list:
            if other != self and self.position.distance_to(other.position) < self.perception_radius:
                steering += other.velocity
                total += 1
        if total > 0:
            steering /= total
            steering = steering.normalize() * self.max_speed
            steering -= self.velocity
            steering = steering.limit(self.max_force)
        return steering

    def cohere(self, fish_list):
        steering = Vector2D(0, 0)
        total = 0
        for other in fish_list:
            if other != self and self.position.distance_to(other.position) < self.perception_radius:
                steering += other.position
                total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            steering = steering.normalize() * self.max_speed
            steering -= self.velocity
            steering = steering.limit(self.max_force)
        return steering

    def separate(self, fish_list):
        steering = Vector2D(0, 0)
        total = 0
        for other in fish_list:
            distance = self.position.distance_to(other.position)
            if other != self and distance < self.avoidance_radius:
                diff = self.position - other.position
                diff /= distance # Weight by distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            steering = steering.normalize() * self.max_speed
            steering -= self.velocity
            steering = steering.limit(self.max_force)
        return steering

    def avoid_shark(self, shark_position):
        steering = Vector2D(0, 0)
        distance = self.position.distance_to(shark_position)
        if distance < self.perception_radius * 2: # Fish perceive shark from further away
            diff = self.position - shark_position
            diff /= distance**2 # Stronger avoidance closer to shark
            steering += diff
            steering = steering.normalize() * self.max_speed
            steering -= self.velocity
            steering = steering.limit(self.max_force * 5) # Very strong avoidance force
        return steering

    def avoid_walls(self):
        steering = Vector2D(0, 0)
        margin = 50
        if self.position.x < margin:
            steering.x = self.max_speed
        elif self.position.x > WIDTH - margin:
            steering.x = -self.max_speed
        if self.position.y < margin:
            steering.y = self.max_speed
        elif self.position.y > HEIGHT - margin:
            steering.y = -self.max_speed
        
        steering = steering.limit(self.max_force)
        return steering

    def draw(self, screen):
        # Draw fish as a triangle pointing in direction of velocity
        angle = math.atan2(self.velocity.y, self.velocity.x)
        p1 = (self.position.x + self.size * math.cos(angle),
              self.position.y + self.size * math.sin(angle))
        p2 = (self.position.x + self.size * math.cos(angle - 2 * math.pi / 3),
              self.position.y + self.size * math.sin(angle - 2 * math.pi / 3))
        p3 = (self.position.x + self.size * math.cos(angle + 2 * math.pi / 3),
              self.position.y + self.size * math.sin(angle + 2 * math.pi / 3))
        pygame.draw.polygon(screen, self.color, [p1, p2, p3])


class Shark:
    def __init__(self, x, y):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * SHARK_SPEED
        self.acceleration = Vector2D(0, 0)
        self.size = SHARK_SIZE
        self.color = (50, 50, 50) # Dark grey
        self.max_speed = SHARK_SPEED
        self.max_force = 0.5
        self.attack_radius = SHARK_ATTACK_RADIUS
        self.is_charging = False
        self.charge_timer = 0

    def apply_force(self, force):
        self.acceleration += force

    def update(self, fish_list, dt):
        # Filter fish that are within the boundaries
        targetable_fish = []
        for fish in fish_list:
            if 0 < fish.position.x < WIDTH and 0 < fish.position.y < HEIGHT:
                targetable_fish.append(fish)

        wall_avoidance = self.avoid_walls()
        self.apply_force(wall_avoidance * 1.0)

        if targetable_fish:
            closest_fish = None
            min_distance = float('inf')
            for fish in targetable_fish:
                distance = self.position.distance_to(fish.position)
                if distance < min_distance:
                    min_distance = distance
                    closest_fish = fish

            if closest_fish:
                # Chase the closest fish
                target = closest_fish.position
                desired_velocity = (target - self.position).normalize() * self.max_speed
                steering = desired_velocity - self.velocity
                steering = steering.limit(self.max_force)
                self.apply_force(steering)

                # Charge at the target if close enough
                if min_distance < 100 and not self.is_charging:
                    self.is_charging = True
                    self.charge_timer = 30 # Charge for 30 frames
        
        # Update charge timer and speed
        if self.is_charging:
            self.max_speed = SHARK_SPEED * 1.5 # Speed boost when charging
            self.charge_timer -= 1
            if self.charge_timer <= 0:
                self.is_charging = False
                self.max_speed = SHARK_SPEED
        else:
            self.max_speed = SHARK_SPEED

        # Update velocity and position using ODEs (Euler integration)
        self.velocity += self.acceleration * dt
        self.velocity = self.velocity.limit(self.max_speed)
        self.position += self.velocity * dt
        self.acceleration *= 0 # Reset acceleration
    def avoid_walls(self):
        steering = Vector2D(0, 0)
        margin = 50
        if self.position.x < margin:
            steering.x = self.max_speed
        elif self.position.x > WIDTH - margin:
            steering.x = -self.max_speed
        if self.position.y < margin:
            steering.y = self.max_speed
        elif self.position.y > HEIGHT - margin:
            steering.y = -self.max_speed
        
        steering = steering.limit(self.max_force)
        return steering

    def draw(self, screen):
        # Draw shark as a triangle pointing in direction of velocity
        angle = math.atan2(self.velocity.y, self.velocity.x)
        p1 = (self.position.x + self.size * 2 * math.cos(angle),
              self.position.y + self.size * 2 * math.sin(angle))
        p2 = (self.position.x + self.size * math.cos(angle - 2.5 * math.pi / 3),
              self.position.y + self.size * math.sin(angle - 2.5 * math.pi / 3))
        p3 = (self.position.x + self.size * math.cos(angle + 2.5 * math.pi / 3),
              self.position.y + self.size * math.sin(angle + 2.5 * math.pi / 3))
        pygame.draw.polygon(screen, self.color, [p1, p2, p3])

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fish Swarm vs Shark (2D)")
    clock = pygame.time.Clock()

    fish_list = [Fish(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) for _ in range(FISH_COUNT)]
    shark = Shark(WIDTH / 2, HEIGHT / 2)

    running = True
    eaten_fish_count = 0
    simulation_time = 0
    dt = (1 / FPS) * TIME_MULTIPLIER # Time step for ODEs

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        # Update and draw fish
        for fish in list(fish_list): # Iterate over a copy to allow removal
            fish.update(fish_list, shark.position, dt)
            fish.draw(screen)

            # Check for shark collision
            if shark.position.distance_to(fish.position) < shark.attack_radius + fish.size:
                fish_list.remove(fish)
                eaten_fish_count += 1
                print(f"Fish eaten! Total: {eaten_fish_count}")

        # Update and draw shark
        shark.update(fish_list, dt)
        shark.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)
        simulation_time += dt

        if not fish_list:
            print("All fish eaten! Simulation ended.")
            running = False

    pygame.quit()
    print(f"Simulation finished after {simulation_time:.2f} seconds.")
    print(f"Total fish eaten: {eaten_fish_count}")
    print(f"Survival rate: {((FISH_COUNT - eaten_fish_count) / FISH_COUNT) * 100:.2f}%")

if __name__ == "__main__":
    run_simulation()
