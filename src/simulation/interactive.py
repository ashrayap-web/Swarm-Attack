"""
Interactive simulation with pygame GUI.
"""

import pygame
import random
import json
import sys
from typing import Optional

from ..core.spatial_grid import SpatialGrid
from ..core.agents.boid import Boid
from ..core.agents.predator import Predator
from ..core.config import SimulationConfig, DEFAULT_CONFIG
from ..hunting.pack import HuntingPack


class Simulation:
    """
    Interactive boids simulation with pygame visualization.
    
    Supports keyboard controls for adjusting simulation parameters
    in real-time and viewing debug information.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation.
        
        Args:
            config: Simulation configuration (uses defaults if None)
        """
        pygame.init()
        
        self.config = config if config else DEFAULT_CONFIG
        self.config_dict = self.config.to_dict()
        
        width = self.config.screenWidth
        height = self.config.screenHeight
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Boids Simulation - Cooperative Hunting")
        self.clock = pygame.time.Clock()
        
        self.spatial_grid = SpatialGrid(width, height, self.config.gridCellSize)
        
        self.boids = []
        self.predators = []
        
        self._spawn_agents()
        
        # Initialize hunting pack
        if self.config.cooperativeHunting and len(self.predators) > 1:
            self.hunting_pack = HuntingPack(self.predators, self.config_dict)
        else:
            self.hunting_pack = None
        
        self.frame_count = 0
        self.running = True
        
        self.stats = {
            "avg_speed": 0,
            "avg_neighbors": 0,
            "avg_stress": 0,
            "flock_cohesion": 0,
            "boids_eaten": 0,
            "formation_quality": 0,
            "pack_state": "none",
            "num_clusters": 0,
            "target_cluster_size": 0
        }
    
    def _spawn_agents(self) -> None:
        """Spawn initial boids and predators."""
        width = self.config.screenWidth
        height = self.config.screenHeight
        
        for _ in range(self.config.boidCount):
            x = random.uniform(100, width - 100)
            y = random.uniform(100, height - 100)
            self.boids.append(Boid(x, y, self.config_dict))
        
        for _ in range(self.config.predatorCount):
            x = random.uniform(50, width - 50)
            y = random.uniform(50, height - 50)
            self.predators.append(Predator(x, y, self.config_dict))
    
    def update(self) -> None:
        """Update simulation state for one frame."""
        # Rebuild spatial grid
        self.spatial_grid.clear()
        for boid in self.boids:
            self.spatial_grid.insert(boid)
        
        # Update boids
        for boid in self.boids:
            boid.flock(self.spatial_grid, self.predators)
            boid.update()
        
        # Update hunting pack coordination
        if self.hunting_pack and self.config.cooperativeHunting:
            current_time = pygame.time.get_ticks()
            self.hunting_pack.update_collective_state(self.boids, current_time)
        
        # Update predators and check for catches
        caught_boids = []
        for predator in self.predators:
            if self.hunting_pack and self.config.cooperativeHunting:
                caught = predator.cooperative_hunt(self.boids, self.predators, self.hunting_pack)
            else:
                caught = predator.hunt(self.boids)
            
            if caught:
                caught_boids.append(caught)
            predator.update()
        
        # Remove caught boids
        for caught in caught_boids:
            if caught in self.boids:
                self.boids.remove(caught)
                self.stats["boids_eaten"] += 1
        
        # Handle split/merge
        if self.config.splitMergeEnabled and self.frame_count % 100 == 0:
            self._handle_split_merge()
        
        self.frame_count += 1
        self._update_statistics()
    
    def _handle_split_merge(self) -> None:
        """Handle population dynamics."""
        if len(self.boids) < self.config.maxAgentCount and random.random() < 0.1:
            x = random.uniform(100, self.config.screenWidth - 100)
            y = random.uniform(100, self.config.screenHeight - 100)
            self.boids.append(Boid(x, y, self.config_dict))
        elif len(self.boids) > self.config.boidCount * 1.5:
            if self.boids:
                self.boids.pop(random.randint(0, len(self.boids) - 1))
    
    def _update_statistics(self) -> None:
        """Update simulation statistics."""
        if not self.boids:
            return
        
        total_speed = sum(b.velocity.length() for b in self.boids)
        self.stats["avg_speed"] = total_speed / len(self.boids)
        
        total_stress = sum(b.stress_level for b in self.boids)
        self.stats["avg_stress"] = total_stress / len(self.boids)
        
        # Flock cohesion
        centroid = pygame.Vector2(0, 0)
        for b in self.boids:
            centroid += b.position
        centroid /= len(self.boids)
        
        total_dist = sum(b.position.distance_to(centroid) for b in self.boids)
        self.stats["flock_cohesion"] = total_dist / len(self.boids)
        
        # Pack statistics
        if self.hunting_pack:
            self.stats["formation_quality"] = self.hunting_pack.formation_quality
            self.stats["pack_state"] = self.hunting_pack.collective_state
            self.stats["num_clusters"] = len(self.hunting_pack.all_clusters)
            if self.hunting_pack.target_cluster:
                self.stats["target_cluster_size"] = self.hunting_pack.target_cluster.size
            else:
                self.stats["target_cluster_size"] = 0
        else:
            self.stats["formation_quality"] = 0
            self.stats["pack_state"] = "none"
            self.stats["num_clusters"] = 0
            self.stats["target_cluster_size"] = 0
    
    def draw(self) -> None:
        """Render the current frame."""
        self.screen.fill(self.config.backgroundColor)
        
        # Draw grid if enabled
        if self.config.visualizeGrid or self.config.visualizationMode == 1:
            self._draw_grid()
        
        # Draw hunting visualization
        debug_mode = self.config.visualizationMode
        if debug_mode >= 2 and self.hunting_pack:
            self._draw_hunting_visualization()
        
        # Draw boids
        for boid in self.boids:
            boid.draw(self.screen, debug_mode)
        
        # Draw predators
        for predator in self.predators:
            formation_pos = None
            if self.hunting_pack and debug_mode >= 2:
                formation_pos = self.hunting_pack.get_formation_position(predator)
            predator.draw(self.screen, debug_mode, formation_pos, self.boids)
        
        self._draw_stats()
        
        pygame.display.flip()
    
    def _draw_hunting_visualization(self) -> None:
        """Draw cooperative hunting visualization."""
        if not self.hunting_pack:
            return
        
        # Draw clusters
        if self.config.enableClustering and self.hunting_pack.all_clusters:
            for cluster in self.hunting_pack.all_clusters:
                color = (80, 80, 80)
                if cluster == self.hunting_pack.target_cluster:
                    color = (100, 200, 100)
                
                pygame.draw.circle(self.screen, color, 
                                 (int(cluster.centroid.x), int(cluster.centroid.y)), 5, 2)
                
                if cluster.radius > 0:
                    pygame.draw.circle(self.screen, color, 
                                     (int(cluster.centroid.x), int(cluster.centroid.y)), 
                                     int(cluster.radius), 1)
                
                font = pygame.font.Font(None, 18)
                text = font.render(str(cluster.size), True, color)
                self.screen.blit(text, (int(cluster.centroid.x) + 10, int(cluster.centroid.y) - 10))
        
        # Draw target centroid
        if self.hunting_pack.prey_centroid:
            centroid = self.hunting_pack.prey_centroid
            pygame.draw.circle(self.screen, (100, 255, 100), 
                             (int(centroid.x), int(centroid.y)), 10, 3)
            pygame.draw.circle(self.screen, (150, 200, 50), 
                             (int(centroid.x), int(centroid.y)), 
                             int(self.config.formationRadius), 2)
    
    def _draw_grid(self) -> None:
        """Draw spatial grid."""
        for i in range(0, self.config.screenWidth, self.config.gridCellSize):
            pygame.draw.line(self.screen, (50, 50, 50), (i, 0), (i, self.config.screenHeight), 1)
        for j in range(0, self.config.screenHeight, self.config.gridCellSize):
            pygame.draw.line(self.screen, (50, 50, 50), (0, j), (self.config.screenWidth, j), 1)
    
    def _draw_stats(self) -> None:
        """Draw statistics overlay."""
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
        
        if self.config.cooperativeHunting and self.hunting_pack:
            stats_text.append(f"Formation: {self.stats['formation_quality']:.2f}")
            stats_text.append(f"Pack: {self.stats['pack_state']}")
            if self.config.enableClustering:
                stats_text.append(f"Clusters: {self.stats['num_clusters']}")
                stats_text.append(f"Target: {self.stats['target_cluster_size']} fish")
        
        for text in stats_text:
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
    
    def save_score(self) -> None:
        """Save simulation score to JSON file."""
        score_data = {
            "frame_count": self.frame_count,
            "boid_count": len(self.boids),
            "predator_count": len(self.predators),
            "statistics": self.stats,
            "config": self.config_dict
        }
        
        try:
            with open(self.config.scoreOutputFile, 'w') as f:
                json.dump(score_data, f, indent=4)
            print(f"Score saved to {self.config.scoreOutputFile}")
        except Exception as e:
            print(f"Error saving score: {e}")
    
    def run(self) -> None:
        """Run the simulation main loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)
            
            self.update()
            self.draw()
            self.clock.tick(self.config.fpsTarget)
        
        self.save_score()
        pygame.quit()
        sys.exit()
    
    def _handle_keydown(self, key: int) -> None:
        """Handle keyboard input."""
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_g:
            self.config.visualizeGrid = not self.config.visualizeGrid
        elif key == pygame.K_v:
            self.config.visualizationMode = (self.config.visualizationMode + 1) % 3
        elif key == pygame.K_c:
            self.config.cooperativeHunting = not self.config.cooperativeHunting
            self.config_dict["cooperativeHunting"] = self.config.cooperativeHunting
            if self.config.cooperativeHunting and len(self.predators) > 1:
                self.hunting_pack = HuntingPack(self.predators, self.config_dict)
            else:
                self.hunting_pack = None
            print(f"Cooperative hunting: {'ON' if self.config.cooperativeHunting else 'OFF'}")
        elif key == pygame.K_l:
            self.config.enableClustering = not self.config.enableClustering
            self.config_dict["enableClustering"] = self.config.enableClustering
            print(f"Smart clustering: {'ON' if self.config.enableClustering else 'OFF'}")
        elif key == pygame.K_t:
            strategies = ["nearest", "largest", "densest"]
            current_idx = strategies.index(self.config.targetStrategy)
            next_idx = (current_idx + 1) % len(strategies)
            self.config.targetStrategy = strategies[next_idx]
            self.config_dict["targetStrategy"] = self.config.targetStrategy
            print(f"Target strategy: {self.config.targetStrategy.upper()}")
        elif key == pygame.K_SPACE:
            self.save_score()

