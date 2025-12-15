"""
Configuration classes and defaults for the boids simulation.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class SimulationConfig:
    """Configuration for the boids simulation."""
    
    # Screen settings
    screenWidth: int = 1200
    screenHeight: int = 650
    
    # Agent counts
    boidCount: int = 100
    predatorCount: int = 5
    maxAgentCount: int = 250
    
    # Movement parameters
    maxSpeed: float = 4.0
    maxForce: float = 0.1
    speedMultiplier: float = 1.0
    accelerationFactor: float = 1.0
    movementNoise: float = 0.01
    
    # Neighbor detection
    neighborRadius: int = 50
    desiredSeparation: int = 20
    gridCellSize: int = 80
    
    # Predator parameters
    predatorAvoidRadius: int = 80
    catchingDistance: int = 8
    predatorInterception: bool = True
    predatorPerception: int = 180
    
    # Predator movement coefficients
    K_C: float = 0.2  # Centering coefficient
    K_D: float = 0.1  # Damping coefficient
    K_B: float = 0.3  # Burst coefficient
    
    # Predator state thresholds
    preyDensityThreshold: int = 20
    burstDuration: int = 1000
    cruiseSpeed: float = 2.0
    burstSpeed: float = 4.0
    
    # Turning constraints
    maxAngularVelocitySlow: float = 0.12
    maxAngularVelocityFast: float = 0.05
    minSpeedForTurning: float = 0.1
    
    # Cooperative hunting
    cooperativeHunting: bool = True
    formationRadius: int = 150
    formationQualityThreshold: float = 0.55
    K_FORMATION: float = 0.25
    K_SEPARATION: float = 0.03
    minPredatorSeparation: int = 80
    compressionForce: float = 0.15
    
    # Smart clustering
    enableClustering: bool = True
    clusterRadius: int = 60
    minClusterSize: int = 5
    targetStrategy: Literal["nearest", "largest", "densest"] = "largest"
    maxTargetDistance: int = 400
    
    # Adaptive behavior
    adaptiveBehaviorEnabled: bool = True
    learningRate: float = 0.05
    splitMergeEnabled: bool = True
    
    # Visualization
    fpsTarget: int = 60
    visualizeGrid: bool = False
    visualizationMode: int = 0  # 0=normal, 1=grid, 2=debug
    backgroundColor: List[int] = field(default_factory=lambda: [25, 25, 25])
    boidColor: List[int] = field(default_factory=lambda: [200, 200, 255])
    predatorColor: List[int] = field(default_factory=lambda: [255, 50, 50])
    
    # Output
    scoreOutputFile: str = "boids_simulation_score.json"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "screenWidth": self.screenWidth,
            "screenHeight": self.screenHeight,
            "boidCount": self.boidCount,
            "predatorCount": self.predatorCount,
            "maxSpeed": self.maxSpeed,
            "maxForce": self.maxForce,
            "neighborRadius": self.neighborRadius,
            "desiredSeparation": self.desiredSeparation,
            "predatorAvoidRadius": self.predatorAvoidRadius,
            "catchingDistance": self.catchingDistance,
            "predatorInterception": self.predatorInterception,
            "gridCellSize": self.gridCellSize,
            "fpsTarget": self.fpsTarget,
            "speedMultiplier": self.speedMultiplier,
            "visualizeGrid": self.visualizeGrid,
            "visualizationMode": self.visualizationMode,
            "backgroundColor": self.backgroundColor,
            "boidColor": self.boidColor,
            "predatorColor": self.predatorColor,
            "maxAgentCount": self.maxAgentCount,
            "accelerationFactor": self.accelerationFactor,
            "movementNoise": self.movementNoise,
            "adaptiveBehaviorEnabled": self.adaptiveBehaviorEnabled,
            "learningRate": self.learningRate,
            "splitMergeEnabled": self.splitMergeEnabled,
            "scoreOutputFile": self.scoreOutputFile,
            "predatorPerception": self.predatorPerception,
            "K_C": self.K_C,
            "K_D": self.K_D,
            "K_B": self.K_B,
            "preyDensityThreshold": self.preyDensityThreshold,
            "burstDuration": self.burstDuration,
            "cruiseSpeed": self.cruiseSpeed,
            "burstSpeed": self.burstSpeed,
            "maxAngularVelocitySlow": self.maxAngularVelocitySlow,
            "maxAngularVelocityFast": self.maxAngularVelocityFast,
            "minSpeedForTurning": self.minSpeedForTurning,
            "cooperativeHunting": self.cooperativeHunting,
            "formationRadius": self.formationRadius,
            "formationQualityThreshold": self.formationQualityThreshold,
            "K_FORMATION": self.K_FORMATION,
            "K_SEPARATION": self.K_SEPARATION,
            "minPredatorSeparation": self.minPredatorSeparation,
            "compressionForce": self.compressionForce,
            "enableClustering": self.enableClustering,
            "clusterRadius": self.clusterRadius,
            "minClusterSize": self.minClusterSize,
            "targetStrategy": self.targetStrategy,
            "maxTargetDistance": self.maxTargetDistance,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# Default configuration for interactive simulation
DEFAULT_CONFIG = SimulationConfig()

# Configuration optimized for benchmarking (no split/merge for consistency)
BENCHMARK_CONFIG = SimulationConfig(
    splitMergeEnabled=False,
    cooperativeHunting=False,
    predatorInterception=False,
    enableClustering=False,
)


# Rule weights (used across modules)
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.3
COHESION_WEIGHT = 1.0
BOUNDARY_WEIGHT = 4.0
PREDATOR_AVOID_WEIGHT = 3.0

# Boundary margin for repulsion
BOUNDARY_MARGIN = 80

