# Simulation Parameters Guide

This document explains all adjustable parameters in the Boids Simulation and how changing them affects the simulation behavior.

## Table of Contents
1. [Boid (Fish) Parameters](#boid-fish-parameters)
2. [Shark/Predator Parameters](#sharkpredator-parameters)
3. [Collision Parameters](#collision-parameters)

---

## Boid (Fish) Parameters

### `NUM_BOIDS` (Default: 150)
**What it does:** Controls the total number of fish in the simulation.

**Impact:**
- **Increase:** More fish = denser swarm, more visual complexity, potentially easier for shark to find prey clusters
- **Decrease:** Fewer fish = sparser swarm, less visual complexity, harder for shark to reach density threshold
- **Note:** Fish are automatically added/removed when this value changes during simulation

---

### `MAX_SPEED` (Default: 2.5)
**What it does:** Maximum velocity magnitude that fish can achieve.

**Impact:**
- **Increase:** Fish move faster, swarm becomes more dynamic and fluid, fish can escape shark more easily
- **Decrease:** Fish move slower, swarm is more sluggish, fish are easier targets for shark
- **Range:** Typically 1.0 - 5.0 for reasonable behavior

---

### `MAX_FORCE` (Default: 0.18)
**What it does:** Maximum steering force that can be applied to change direction.

**Impact:**
- **Increase:** Fish can turn more sharply and react more quickly to threats, more agile movement
- **Decrease:** Fish turn more gradually, smoother but less responsive movement
- **Note:** Works in conjunction with MAX_SPEED - higher force allows faster acceleration

---

### `PERCEPTION_RADIUS` (Default: 55)
**What it does:** Distance within which fish can "see" and react to neighbors and the shark.

**Impact:**
- **Increase:** Fish react to neighbors/shark from further away, larger flocking groups, more coordinated behavior
- **Decrease:** Fish only react to nearby neighbors, smaller local groups, more fragmented swarm
- **Note:** Affects all three flocking behaviors (separation, alignment, cohesion)

---

### `SEPARATION_WEIGHT` (Default: 0.8)
**What it does:** Strength of the separation force - fish avoid crowding each other.

**Impact:**
- **Increase:** Fish maintain more distance from each other, less clumping, more spread out swarm
- **Decrease:** Fish allow closer proximity, more compact groups, tighter formations
- **Range:** 0.0 (no separation) to 2.0+ (strong separation)

---

### `ALIGNMENT_WEIGHT` (Default: 0.8)
**What it does:** Strength of the alignment force - fish try to match velocity direction of neighbors.

**Impact:**
- **Increase:** Fish move in more coordinated directions, stronger schooling behavior, more synchronized movement
- **Decrease:** Fish move more independently, less coordinated, more chaotic movement
- **Range:** 0.0 (no alignment) to 2.0+ (strong alignment)

---

### `COHESION_WEIGHT` (Default: 0.7)
**What it does:** Strength of the cohesion force - fish are attracted to the center of nearby neighbors.

**Impact:**
- **Increase:** Fish form tighter groups, stronger clustering, more compact swarm
- **Decrease:** Fish are less attracted to groups, more dispersed, looser formations
- **Range:** 0.0 (no cohesion) to 2.0+ (strong cohesion)

---

### `PREDATOR_EVADE_WEIGHT` (Default: 2.5)
**What it does:** Strength of the evasion force - how strongly fish flee from the shark.

**Impact:**
- **Increase:** Fish react more strongly to shark, flee faster and from further away, harder for shark to catch them
- **Decrease:** Fish react less to shark, slower evasion, easier for shark to approach
- **Range:** 0.0 (ignore shark) to 5.0+ (extreme panic)

---

## Shark/Predator Parameters

### `PREDATOR_PERCEPTION` (Default: 120)
**What it does:** Distance within which the shark can detect and track fish.

**Impact:**
- **Increase:** Shark can see fish from further away, starts herding earlier, more effective hunting
- **Decrease:** Shark only detects nearby fish, must get closer to start herding, less effective
- **Note:** This is the radius used for calculating prey density and centroid

---

### `K_C` (Default: 0.15)
**What it does:** Centering force coefficient in herding mode - attraction strength toward prey centroid.

**Impact:**
- **Increase:** Shark is more strongly attracted to prey center, more aggressive herding, faster compaction
- **Decrease:** Shark is less attracted, gentler herding, slower compaction
- **Formula:** Part of `a_s = k_c(x_prey_centroid - x_s) - k_d*v_s` in herding mode

---

### `K_D` (Default: 0.08)
**What it does:** Damping coefficient in herding mode - opposes velocity to create circling behavior.

**Impact:**
- **Increase:** More damping, shark slows down more, tighter circling, better herding behavior
- **Decrease:** Less damping, shark maintains more speed, wider circles, less effective herding
- **Formula:** Part of `a_s = k_c(x_prey_centroid - x_s) - k_d*v_s` in herding mode

---

### `K_B` (Default: 0.35)
**What it does:** Burst acceleration coefficient - how strongly shark accelerates toward prey during burst.

**Impact:**
- **Increase:** Stronger burst acceleration, faster attacks, more aggressive bursts
- **Decrease:** Weaker burst acceleration, slower attacks, less aggressive bursts
- **Formula:** `a_s = k_b(x_prey_centroid - x_s)` in burst mode

---

### `PREY_DENSITY_THRESHOLD` (Default: 25)
**What it does:** Minimum number of fish within perception radius needed to trigger burst mode.

**Impact:**
- **Increase:** Shark needs more fish nearby to burst, stays in herding longer, more patient hunting
- **Decrease:** Shark bursts with fewer fish nearby, transitions to burst sooner, more aggressive
- **Note:** Lower values = more frequent bursts, higher values = more herding before attack

---

### `BURST_DURATION` (Default: 800 milliseconds)
**What it does:** How long the shark stays in burst mode before returning to herding.

**Impact:**
- **Increase:** Longer bursts, more time to catch fish, but longer recovery period
- **Decrease:** Shorter bursts, less time to catch fish, but faster return to herding
- **Range:** Typically 200-2000 milliseconds

---

### `CRUISE_SPEED` (Default: 1.5)
**What it does:** Maximum speed of shark while in herding/cruising mode.

**Impact:**
- **Increase:** Faster herding, quicker movement around prey, faster compaction
- **Decrease:** Slower herding, more deliberate movement, slower compaction
- **Note:** Should be lower than BURST_SPEED for realistic behavior

---

### `BURST_SPEED` (Default: 4.0)
**What it does:** Maximum speed of shark during burst/attack mode.

**Impact:**
- **Increase:** Faster attacks, harder for fish to escape, more successful catches
- **Decrease:** Slower attacks, easier for fish to escape, less successful catches
- **Note:** Should be higher than CRUISE_SPEED for realistic burst behavior

---

### `MAX_ANGULAR_VELOCITY_SLOW` (Default: 0.08 radians/frame)
**What it does:** Maximum turning rate when shark is moving slowly (realistic movement constraint).

**Impact:**
- **Increase:** Shark can turn more sharply at slow speeds, more agile during herding
- **Decrease:** Shark turns more gradually at slow speeds, more realistic but less agile
- **Note:** This creates realistic inertia - faster movement = less turning ability

---

### `MAX_ANGULAR_VELOCITY_FAST` (Default: 0.03 radians/frame)
**What it does:** Maximum turning rate when shark is moving fast (realistic movement constraint).

**Impact:**
- **Increase:** Shark can turn more sharply at high speeds, less realistic but more agile
- **Decrease:** Shark turns more gradually at high speeds, more realistic inertia during bursts
- **Note:** Should be lower than MAX_ANGULAR_VELOCITY_SLOW for realistic behavior

---

### `MIN_SPEED_FOR_TURNING` (Default: 0.1)
**What it does:** Speed threshold below which turning constraints don't apply (allows free turning when nearly stopped).

**Impact:**
- **Increase:** Shark has more freedom to turn at low speeds, easier to change direction
- **Decrease:** Turning constraints apply at lower speeds, more realistic but less maneuverable
- **Note:** Prevents shark from getting "stuck" when moving very slowly

---

## Collision Parameters

### `COLLISION_RADIUS` (Default: 15)
**What it does:** Distance at which shark "eats" a fish (collision detection radius).

**Impact:**
- **Increase:** Easier for shark to catch fish, larger "hitbox", more successful attacks
- **Decrease:** Harder for shark to catch fish, smaller "hitbox", requires more precision
- **Range:** Typically 5-30 pixels for reasonable gameplay

---

## Parameter Interaction Effects

### Creating Realistic Herding Behavior
- **High K_C + High K_D:** Strong herding with tight circling
- **Low K_C + Low K_D:** Gentle herding with wide circles

### Creating Effective Burst Attacks
- **High K_B + High BURST_SPEED:** Fast, aggressive attacks
- **Low PREY_DENSITY_THRESHOLD:** More frequent bursts
- **High BURST_DURATION:** Longer attack windows

### Creating Realistic Movement
- **Low MAX_ANGULAR_VELOCITY_FAST:** Realistic inertia at high speeds
- **High MAX_ANGULAR_VELOCITY_SLOW:** Agile movement during herding

### Balancing Fish Survival
- **High PREDATOR_EVADE_WEIGHT + High MAX_SPEED:** Fish escape more easily
- **Low COLLISION_RADIUS:** Harder for shark to catch fish
- **High SEPARATION_WEIGHT:** Fish spread out, harder to herd

---

## Quick Tuning Tips

**Make shark more effective:**
- Increase `PREDATOR_PERCEPTION`
- Decrease `PREY_DENSITY_THRESHOLD`
- Increase `COLLISION_RADIUS`
- Increase `BURST_SPEED` and `K_B`

**Make fish harder to catch:**
- Increase `PREDATOR_EVADE_WEIGHT`
- Increase `MAX_SPEED` (fish)
- Increase `SEPARATION_WEIGHT`
- Decrease `COLLISION_RADIUS`

**Create more dramatic herding:**
- Increase `K_C` and `K_D`
- Increase `CRUISE_SPEED`
- Increase `PREY_DENSITY_THRESHOLD` (more herding before burst)

**Make movement more realistic:**
- Decrease `MAX_ANGULAR_VELOCITY_FAST`
- Keep `BURST_SPEED` significantly higher than `CRUISE_SPEED`

