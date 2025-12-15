"""
Benchmark simulation for performance testing and data collection.
"""

import pygame
import random
import time
from typing import Dict, List, Optional, Any

try:
    import numpy as np
    import cv2
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

from ..core.spatial_grid import SpatialGrid
from ..core.agents.boid import Boid
from ..core.agents.predator import Predator
from ..hunting.pack import HuntingPack


# Default benchmark configuration
COHESION_TRACKING_INTERVAL = 10


class BenchmarkSimulation:
    """
    Benchmark simulation for testing hunting method performance.
    
    Runs headless (no GUI) by default but supports video recording.
    Collects detailed statistics for analysis.
    """
    
    def __init__(self, config: Dict, enable_video: bool = False, 
                 video_filename: Optional[str] = None, video_fps: int = 30):
        """
        Initialize benchmark simulation.
        
        Args:
            config: Configuration dictionary
            enable_video: Whether to record video
            video_filename: Output video filename
            video_fps: Video frame rate
        """
        pygame.init()
        
        self.config = config
        width = config["screenWidth"]
        height = config["screenHeight"]
        
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # Video recording
        self.enable_video = enable_video and VIDEO_SUPPORT
        self.video_writer = None
        self.video_filename = video_filename
        self.frame_skip = max(1, config.get("fpsTarget", 60) // video_fps)
        
        if self.enable_video and self.video_filename:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, video_fps, (width, height)
            )
            print(f"  Recording video to: {self.video_filename}")
        
        self.spatial_grid = SpatialGrid(width, height, config["gridCellSize"])
        self.boids = []
        self.predators = []
        
        self._spawn_agents()
        
        if config.get("cooperativeHunting", False) and len(self.predators) > 1:
            self.hunting_pack = HuntingPack(self.predators, config)
        else:
            self.hunting_pack = None
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # Statistics
        self.stats = {
            "boids_eaten": 0,
            "total_distance_traveled": 0,
            "burst_attempts": 0,
            "successful_bursts": 0,
            "burst_captures": 0,
            "herding_captures": 0,
            "time_in_burst": 0,
            "time_in_herding": 0,
            "avg_formation_quality": 0,
            "formation_quality_samples": 0,
            "avg_stress": 0,
            "avg_cohesion": 0,
            "avg_predator_speed": 0,
            "predator_speed_samples": 0,
            "captures_per_1000_frames": [],
            "first_capture_frame": None,
            "time_between_captures": [],
            "last_capture_frame": 0,
            "cumulative_captures_over_time": [],
            "cohesion_over_time": [],
        }
        
        self.previous_pack_state = None
    
    def _spawn_agents(self) -> None:
        """Spawn initial agents."""
        width = self.config["screenWidth"]
        height = self.config["screenHeight"]
        
        for _ in range(self.config["boidCount"]):
            x = random.uniform(100, width - 100)
            y = random.uniform(100, height - 100)
            self.boids.append(Boid(x, y, self.config))
        
        for _ in range(self.config["predatorCount"]):
            x = random.uniform(50, width - 50)
            y = random.uniform(50, height - 50)
            self.predators.append(Predator(x, y, self.config))
    
    def update(self) -> None:
        """Update simulation state."""
        self.spatial_grid.clear()
        for boid in self.boids:
            self.spatial_grid.insert(boid)
        
        for boid in self.boids:
            boid.flock(self.spatial_grid, self.predators)
            boid.update()
        
        if self.hunting_pack and self.config.get("cooperativeHunting", False):
            current_time = pygame.time.get_ticks()
            self.hunting_pack.update_collective_state(self.boids, current_time)
            
            # Track burst attempts
            if self.previous_pack_state == "herding" and self.hunting_pack.collective_state == "burst":
                self.stats["burst_attempts"] += 1
            
            self.previous_pack_state = self.hunting_pack.collective_state
        
        caught_boids = []
        for predator in self.predators:
            if self.hunting_pack and self.config.get("cooperativeHunting", False):
                caught = predator.cooperative_hunt(self.boids, self.predators, self.hunting_pack)
            else:
                caught = predator.hunt(self.boids)
            
            if caught:
                caught_boids.append(caught)
                if predator.state == "burst":
                    self.stats["burst_captures"] += 1
                else:
                    self.stats["herding_captures"] += 1
            
            predator.update()
        
        # Remove caught boids
        for caught in caught_boids:
            if caught in self.boids:
                self.boids.remove(caught)
                self.stats["boids_eaten"] += 1
                
                if self.stats["first_capture_frame"] is None:
                    self.stats["first_capture_frame"] = self.frame_count
                
                frames_since_last = self.frame_count - self.stats["last_capture_frame"]
                if self.stats["last_capture_frame"] > 0:
                    self.stats["time_between_captures"].append(frames_since_last)
                
                self.stats["last_capture_frame"] = self.frame_count
        
        if caught_boids and self.hunting_pack and self.hunting_pack.collective_state == "burst":
            self.stats["successful_bursts"] += 1
        
        self.frame_count += 1
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update tracking statistics."""
        # Distance traveled
        total_dist = sum(pred.distance_traveled for pred in self.predators)
        self.stats["total_distance_traveled"] = total_dist
        
        # State time tracking
        if self.hunting_pack:
            if self.hunting_pack.collective_state == "burst":
                self.stats["time_in_burst"] += 1
            else:
                self.stats["time_in_herding"] += 1
            
            self.stats["avg_formation_quality"] += self.hunting_pack.formation_quality
            self.stats["formation_quality_samples"] += 1
        
        # Predator speed
        if self.predators:
            total_speed = sum(pred.velocity.length() for pred in self.predators)
            self.stats["avg_predator_speed"] += total_speed / len(self.predators)
            self.stats["predator_speed_samples"] += 1
        
        # Prey behavior
        if self.boids:
            total_stress = sum(b.stress_level for b in self.boids)
            self.stats["avg_stress"] = total_stress / len(self.boids)
            
            centroid = pygame.Vector2(0, 0)
            for b in self.boids:
                centroid += b.position
            centroid /= len(self.boids)
            
            total_dist = sum(b.position.distance_to(centroid) for b in self.boids)
            current_cohesion = total_dist / len(self.boids)
            self.stats["avg_cohesion"] = current_cohesion
            
            # Time-series cohesion
            if self.frame_count % COHESION_TRACKING_INTERVAL == 0:
                self.stats["cohesion_over_time"].append({
                    "frame": self.frame_count,
                    "cohesion": current_cohesion,
                    "boid_count": len(self.boids)
                })
        
        # Captures per 1000 frames
        if self.frame_count % 1000 == 0:
            recent = sum(1 for t in self.stats["time_between_captures"] 
                        if self.frame_count - t <= 1000)
            self.stats["captures_per_1000_frames"].append(recent)
        
        # Cumulative captures
        if self.frame_count % 10 == 0:
            self.stats["cumulative_captures_over_time"].append({
                "frame": self.frame_count,
                "captures": self.stats["boids_eaten"]
            })
    
    def run_benchmark(self, max_frames: int) -> Dict[str, Any]:
        """
        Run benchmark for specified number of frames.
        
        Args:
            max_frames: Maximum frames to simulate
            
        Returns:
            Results dictionary with all statistics
        """
        print(f"Running benchmark for {max_frames} frames...")
        
        while self.frame_count < max_frames:
            self.update()
            
            if self.enable_video and self.video_writer:
                if self.frame_count % self.frame_skip == 0:
                    self._render_frame()
                    self._capture_frame()
            
            if self.frame_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                progress = (self.frame_count / max_frames) * 100
                print(f"  Progress: {progress:.1f}% ({self.frame_count}/{max_frames} frames, "
                      f"{elapsed:.1f}s elapsed, {self.stats['boids_eaten']} captures)")
        
        if self.video_writer:
            self.video_writer.release()
            print("  Video saved successfully!")
        
        pygame.quit()
        return self.get_results()
    
    def _render_frame(self) -> None:
        """Render frame for video capture."""
        self.screen.fill(self.config["backgroundColor"])
        
        # Draw hunting visualization
        if self.hunting_pack:
            self._draw_hunting_visualization()
        
        # Draw boids
        for boid in self.boids:
            color = self.config["boidColor"]
            stress_color = int(255 * boid.stress_level)
            color = (stress_color, color[1], color[2])
            pygame.draw.circle(self.screen, color, 
                             (int(boid.position.x), int(boid.position.y)), 4)
            if boid.velocity.length() > 0:
                end_pos = boid.position + boid.velocity.normalize() * 10
                pygame.draw.line(self.screen, color, boid.position, end_pos, 1)
        
        # Draw predators
        for predator in self.predators:
            formation_pos = None
            if self.hunting_pack:
                formation_pos = self.hunting_pack.get_formation_position(predator)
            self._draw_predator(predator, formation_pos)
        
        self._draw_stats_overlay()
        pygame.display.flip()
    
    def _draw_predator(self, predator, formation_pos) -> None:
        """Draw a predator."""
        if predator.state == "burst":
            color = (255, 100, 100)
        else:
            color = self.config["predatorColor"]
        
        pygame.draw.circle(self.screen, color, 
                         (int(predator.position.x), int(predator.position.y)), 6)
        pygame.draw.circle(self.screen, (200, 0, 0), 
                         (int(predator.position.x), int(predator.position.y)), 6, 2)
        
        # Perception radius
        pygame.draw.circle(self.screen, (100, 50, 50), 
                         (int(predator.position.x), int(predator.position.y)), 
                         int(predator.perception), 1)
        
        # Formation target
        if formation_pos and predator.state == "herding":
            pygame.draw.circle(self.screen, (255, 255, 0), 
                             (int(formation_pos.x), int(formation_pos.y)), 5, 2)
            pygame.draw.line(self.screen, (255, 255, 0), predator.position, formation_pos, 1)
        
        # Interception point
        if predator.state == "burst" and self.boids and self.config.get("predatorInterception", True):
            closest = min(self.boids, key=lambda b: predator.position.distance_to(b.position))
            intercept = predator.calculate_interception(closest)
            pygame.draw.line(self.screen, (255, 150, 0), predator.position, intercept, 2)
            size = 5
            pygame.draw.line(self.screen, (255, 200, 0), 
                           (intercept.x - size, intercept.y - size),
                           (intercept.x + size, intercept.y + size), 2)
            pygame.draw.line(self.screen, (255, 200, 0),
                           (intercept.x + size, intercept.y - size),
                           (intercept.x - size, intercept.y + size), 2)
    
    def _draw_hunting_visualization(self) -> None:
        """Draw hunting pack visualization."""
        if not self.hunting_pack:
            return
        
        if self.config.get("enableClustering", False) and self.hunting_pack.all_clusters:
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
        
        if self.hunting_pack.prey_centroid:
            centroid = self.hunting_pack.prey_centroid
            pygame.draw.circle(self.screen, (100, 255, 100), 
                             (int(centroid.x), int(centroid.y)), 10, 3)
            pygame.draw.circle(self.screen, (150, 200, 50), 
                             (int(centroid.x), int(centroid.y)), 
                             int(self.config['formationRadius']), 2)
    
    def _draw_stats_overlay(self) -> None:
        """Draw stats on screen."""
        font = pygame.font.Font(None, 24)
        y = 10
        
        texts = [
            f"Frame: {self.frame_count}",
            f"Boids: {len(self.boids)}",
            f"Eaten: {self.stats['boids_eaten']}",
            f"Avg Stress: {self.stats['avg_stress']:.2f}",
            f"Cohesion: {self.stats['avg_cohesion']:.1f}"
        ]
        
        if self.config.get("cooperativeHunting", False) and self.hunting_pack:
            texts.append(f"Formation: {self.hunting_pack.formation_quality:.2f}")
            texts.append(f"Pack: {self.hunting_pack.collective_state}")
            if self.config.get("enableClustering", False):
                texts.append(f"Clusters: {len(self.hunting_pack.all_clusters)}")
        
        for text in texts:
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, y))
            y += 25
    
    def _capture_frame(self) -> None:
        """Capture frame to video."""
        if not self.video_writer or not VIDEO_SUPPORT:
            return
        
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get benchmark results.
        
        Returns:
            Dictionary containing all statistics and derived metrics
        """
        elapsed = time.time() - self.start_time
        
        avg_formation = 0
        if self.stats["formation_quality_samples"] > 0:
            avg_formation = self.stats["avg_formation_quality"] / self.stats["formation_quality_samples"]
        
        avg_between = 0
        if self.stats["time_between_captures"]:
            avg_between = sum(self.stats["time_between_captures"]) / len(self.stats["time_between_captures"])
        
        burst_rate = 0
        if self.stats["burst_attempts"] > 0:
            burst_rate = self.stats["successful_bursts"] / self.stats["burst_attempts"]
        
        captures_per_dist = 0
        if self.stats["total_distance_traveled"] > 0:
            captures_per_dist = self.stats["boids_eaten"] / self.stats["total_distance_traveled"]
        
        captures_per_frame = 0
        if self.frame_count > 0:
            captures_per_frame = self.stats["boids_eaten"] / self.frame_count
        
        avg_speed = 0
        if self.stats["predator_speed_samples"] > 0:
            avg_speed = self.stats["avg_predator_speed"] / self.stats["predator_speed_samples"]
        
        return {
            "total_captures": self.stats["boids_eaten"],
            "frames": self.frame_count,
            "elapsed_time_seconds": elapsed,
            "total_distance_traveled": self.stats["total_distance_traveled"],
            "burst_captures": self.stats["burst_captures"],
            "herding_captures": self.stats["herding_captures"],
            "burst_attempts": self.stats["burst_attempts"],
            "successful_bursts": self.stats["successful_bursts"],
            "burst_success_rate": burst_rate,
            "time_in_burst": self.stats["time_in_burst"],
            "time_in_herding": self.stats["time_in_herding"],
            "avg_formation_quality": avg_formation,
            "avg_stress": self.stats["avg_stress"],
            "avg_cohesion": self.stats["avg_cohesion"],
            "avg_predator_speed": avg_speed,
            "first_capture_frame": self.stats["first_capture_frame"],
            "avg_time_between_captures": avg_between,
            "captures_per_distance": captures_per_dist,
            "captures_per_frame": captures_per_frame,
            "captures_per_1000_frames": self.stats["captures_per_1000_frames"],
            "final_prey_count": len(self.boids),
            "cumulative_captures_over_time": self.stats["cumulative_captures_over_time"],
            "cohesion_over_time": self.stats["cohesion_over_time"],
        }

