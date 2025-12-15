# Boids Simulation with Cooperative Predator Hunting

A sophisticated simulation modeling boid flocking behavior and predator-prey dynamics with advanced cooperative hunting strategies.

## Overview

This simulation implements Craig Reynolds' classic boids algorithm for modeling flocking behavior, extended with:

- **Predator-Prey Dynamics**: Predators hunt prey using realistic herding and burst strategies
- **Cooperative Hunting**: Multiple predators coordinate to encircle and capture prey
- **Smart Clustering**: Intelligent detection of prey groups for targeted hunting
- **Predictive Interception**: Predators can predict prey movement for more efficient captures

## Project Structure

```
src/
├── main.py                      # Entry point and CLI
├── core/                        # Core simulation components
│   ├── config.py               # Configuration classes and defaults
│   ├── spatial_grid.py         # Spatial hash grid for neighbor lookup
│   └── agents/                 # Agent implementations
│       ├── base.py            # Base Agent class
│       ├── boid.py            # Boid (prey) with flocking behavior
│       └── predator.py        # Predator with hunting behavior
├── hunting/                     # Hunting coordination
│   ├── clustering.py           # Prey cluster detection
│   └── pack.py                 # HuntingPack coordination
├── simulation/                  # Simulation runners
│   ├── interactive.py          # Interactive GUI simulation
│   └── benchmark.py            # Headless benchmark simulation
└── analysis/                    # Results analysis
    ├── plotting.py             # Visualization and plots
    └── export.py               # CSV/JSON export functions
```

## Requirements

### Core Dependencies
- **Python 3.8+**
- **pygame** - Game library for visualization and simulation

### Optional Dependencies (for benchmarking/analysis)
- **numpy** - Numerical operations
- **opencv-python** - Video recording
- **matplotlib** - Plot generation

Install all dependencies:
```bash
pip install pygame numpy opencv-python matplotlib
```

## Usage

### Interactive Simulation

Run the visual simulation with keyboard controls:

```bash
python -m src.main
```

**Keyboard Controls:**
| Key | Action |
|-----|--------|
| `ESC` | Quit simulation |
| `G` | Toggle grid visualization |
| `V` | Cycle visualization modes (normal → grid → debug) |
| `C` | Toggle cooperative hunting ON/OFF |
| `L` | Toggle smart clustering ON/OFF |
| `T` | Cycle target strategy (nearest → largest → densest) |
| `SPACE` | Save score to JSON |

### Benchmark Mode

Run headless benchmarks comparing hunting methods:

```bash
python -m src.main --benchmark
```

Options:
- `--trials N` - Number of trials per method (default: 50)
- `--duration N` - Frames per trial (default: 5000)
- `--record-video` - Record video of one trial per method

### Cohesion Analysis

Run cohesion time-series analysis comparing models:

```bash
python -m src.main --cohesion
```

## Hunting Methods Compared

### Method 1: Basic Herding
- Individual predators chase prey groups
- No predictive interception
- Simple state machine: herding → burst → herding

### Method 2: Herding + Interception
- Individual predators with predictive targeting
- Calculates interception point based on prey velocity
- More efficient captures through anticipation

### Method 3: Full Coordination (Cooperative Hunting)
- Pack-based hunting with formation maintenance
- Predators position themselves around prey groups
- Coordinated burst attacks when formation quality is high
- Smart clustering identifies distinct prey groups

## Algorithm Details

### Boid Flocking (Reynolds' Rules)

1. **Separation**: Steer away from nearby flockmates
2. **Alignment**: Steer toward average heading of neighbors
3. **Cohesion**: Steer toward average position of neighbors

Plus:
- **Boundary Repulsion**: Stay within simulation bounds
- **Predator Avoidance**: Flee from nearby predators
- **Adaptive Behavior**: Adjust weights based on stress level

### Predator Hunting

**Herding Phase:**
```
acceleration = K_C * (prey_centroid - position) - K_D * velocity
```

**Burst Phase:**
- Direct pursuit or predictive interception
- Higher maximum speed
- Limited duration before returning to herding

### Cooperative Hunting

1. **Formation Assignment**: Predators assigned angular positions around prey
2. **Formation Quality**: Measured by how evenly distributed predators are
3. **Burst Trigger**: When formation quality ≥ threshold AND prey density ≥ threshold
4. **Cluster Targeting**: Smart selection of prey groups based on strategy

### Predictive Interception

The interception point is calculated by:
1. Computing relative velocity between predator and prey
2. Estimating time-to-intercept based on closing speed
3. Predicting prey position at intercept time
4. Steering toward predicted position

## Configuration

Key configuration parameters in `core/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `boidCount` | 100 | Number of prey |
| `predatorCount` | 5 | Number of predators |
| `maxSpeed` | 4.0 | Max prey speed |
| `cruiseSpeed` | 2.0 | Predator herding speed |
| `burstSpeed` | 4.0 | Predator burst speed |
| `formationRadius` | 150 | Formation distance from prey |
| `formationQualityThreshold` | 0.55 | Min quality to trigger burst |
| `clusterRadius` | 60 | Max distance within cluster |
| `targetStrategy` | "largest" | Cluster selection strategy |

## Output Files

### Interactive Mode
- `boids_simulation_score.json` - Final simulation statistics

### Benchmark Mode
- `benchmark_results.csv` - Per-trial results
- `hunting_benchmark_results.json` - Full benchmark report
- `hunting_performance_comparison.png` - Captures over time plot
- `cohesion_timeseries_N_predators.csv` - Cohesion data for N predators
- `cohesion_comparison_all_predators.png` - Cohesion comparison plot

## Visualization Modes

### Mode 0: Normal
- Boids shown as blue circles
- Predators shown as red circles (brighter during burst)

### Mode 1: Grid
- Shows spatial hash grid overlay
- Helps visualize neighbor detection regions

### Mode 2: Debug
- Boid color indicates stress level (red = high stress)
- Predator perception radius shown
- Formation target positions visible
- Interception points marked during burst
- Cluster boundaries and sizes displayed

## Research Applications

This simulation is suitable for studying:

- **Collective Behavior**: How local rules produce emergent group dynamics
- **Predator-Prey Dynamics**: Effectiveness of different hunting strategies
- **Cooperative Strategies**: Benefits of coordination vs. individual hunting
- **Bait Ball Formation**: How prey clustering changes under predation pressure
- **Optimal Foraging**: Trade-offs between energy expenditure and capture success

## License

This code is provided for educational and research purposes.

## References

1. Reynolds, C. W. (1987). "Flocks, herds and schools: A distributed behavioral model"
2. Handegard, N. O., et al. (2012). "Dynamics of coordinated group hunting and collective information transfer"
3. Pitcher, T. J., & Parrish, J. K. (1993). "Functions of shoaling behaviour in teleosts"

