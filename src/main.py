"""
Main entry point for the boids simulation.

Run with:
    python -m src.main              # Interactive simulation
    python -m src.main --benchmark  # Run benchmarks
    python -m src.main --cohesion   # Cohesion analysis only
"""

import os
import sys
import random
import math
import json

# Set dummy video driver for headless benchmarking
def set_headless():
    """Enable headless mode for benchmarking."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def run_interactive():
    """Run the interactive simulation with GUI."""
    from .simulation.interactive import Simulation
    from .core.config import SimulationConfig
    
    print("=" * 60)
    print("Boids Simulation with Cooperative Hunting & Smart Clustering")
    print("=" * 60)
    print("\nControls:")
    print("  ESC   - Quit")
    print("  G     - Toggle grid visualization")
    print("  V     - Cycle visualization modes (0=normal, 1=grid, 2=debug)")
    print("  C     - Toggle cooperative hunting ON/OFF")
    print("  L     - Toggle smart clustering ON/OFF")
    print("  T     - Cycle target strategy (nearest/largest/densest)")
    print("  SPACE - Save score to JSON")
    print("\nCooperative Hunting:")
    print("  When enabled, predators work together to encircle prey")
    print("  They position themselves around the group before bursting")
    print("  Watch the 'Formation' stat to see coordination quality")
    print("\nSmart Clustering:")
    print("  Detects actual groups of fish instead of using global centroid")
    print("  Prevents predators from targeting empty space between split groups")
    print("  Press V twice to see clusters visualized (green = target)")
    print("\nStarting simulation...")
    
    config = SimulationConfig()
    sim = Simulation(config)
    sim.run()


def run_benchmark(num_trials: int = 50, duration: int = 5000, 
                  record_video: bool = False, video_trial: int = 1):
    """
    Run full benchmark comparison of hunting methods.
    
    Args:
        num_trials: Number of trials per method
        duration: Duration in frames per trial
        record_video: Whether to record video
        video_trial: Which trial to record
    """
    set_headless()
    
    from .simulation.benchmark import BenchmarkSimulation
    from .analysis.export import export_results_to_csv, calculate_aggregate_stats, export_benchmark_report
    from .analysis.plotting import plot_cumulative_captures
    from .core.config import SimulationConfig
    
    print("=" * 60)
    print("HUNTING METHOD BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"Duration per trial: {duration} frames")
    print(f"Trials per configuration: {num_trials}")
    print(f"Total simulations: {num_trials * 3}")
    if record_video:
        print(f"Video recording: ENABLED (trial {video_trial})")
    print()
    
    base_config = SimulationConfig().to_dict()
    base_config["splitMergeEnabled"] = False  # Consistent benchmarking
    
    # Run configurations
    def run_config(name, overrides, method_id):
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {name}")
        print(f"{'=' * 60}")
        
        results = []
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}")
            
            config = base_config.copy()
            config.update(overrides)
            
            random.seed(42 + trial)
            
            enable_video = record_video and (trial + 1) == video_trial
            video_file = None
            if enable_video:
                safe_name = name.lower().replace(" ", "_").replace(":", "").replace("+", "and")
                video_file = f"recording_method{method_id}_{safe_name}_trial{trial + 1}.mp4"
            
            sim = BenchmarkSimulation(config, enable_video=enable_video, video_filename=video_file)
            result = sim.run_benchmark(duration)
            result["trial"] = trial + 1
            if enable_video:
                result["video_file"] = video_file
            results.append(result)
        
        return results
    
    # Method 1: Basic Herding
    config1 = {"cooperativeHunting": False, "predatorInterception": False, "enableClustering": False}
    results1 = run_config("Method 1: Basic Herding", config1, 1)
    
    # Method 2: Herding + Interception
    config2 = {"cooperativeHunting": False, "predatorInterception": True, "enableClustering": False}
    results2 = run_config("Method 2: Herding + Interception", config2, 2)
    
    # Method 3: Full Coordination
    config3 = {"cooperativeHunting": True, "predatorInterception": True, "enableClustering": True}
    results3 = run_config("Method 3: Full Coordination", config3, 3)
    
    # Calculate aggregates
    agg1 = calculate_aggregate_stats(results1)
    agg2 = calculate_aggregate_stats(results2)
    agg3 = calculate_aggregate_stats(results3)
    
    # Determine winners
    methods = [("Basic Herding", agg1), ("Herding + Interception", agg2), ("Full Coordination", agg3)]
    
    best_captures = max(methods, key=lambda x: x[1].get("total_captures_mean", 0))
    best_efficiency = max(methods, key=lambda x: x[1].get("captures_per_distance_mean", 0))
    best_first = min(methods, key=lambda x: x[1].get("first_capture_frame_mean", float('inf')))
    
    # Prepare report
    report = {
        "benchmark_config": {"duration_frames": duration, "trials_per_method": num_trials},
        "method_1_basic_herding": {"config": config1, "trial_results": results1, "aggregates": agg1},
        "method_2_herding_interception": {"config": config2, "trial_results": results2, "aggregates": agg2},
        "method_3_full_coordination": {"config": config3, "trial_results": results3, "aggregates": agg3},
        "comparison": {
            "winner_by_total_captures": best_captures[0],
            "winner_by_efficiency": best_efficiency[0],
            "winner_by_time_to_first": best_first[0],
        }
    }
    
    # Save results
    export_benchmark_report(report)
    export_results_to_csv(results1, results2, results3)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    for name, agg in methods:
        print(f"\n{name.upper()}:")
        print(f"   Total Captures: {agg.get('total_captures_mean', 0):.2f} ± {agg.get('total_captures_std', 0):.2f}")
        print(f"   Efficiency: {agg.get('captures_per_distance_mean', 0):.6f}")
        print(f"   First Capture: {agg.get('first_capture_frame_mean', 0):.0f} frames")
    
    print("\n" + "=" * 60)
    print("WINNERS:")
    print("=" * 60)
    print(f"Most Captures: {report['comparison']['winner_by_total_captures']}")
    print(f"Best Efficiency: {report['comparison']['winner_by_efficiency']}")
    print(f"Fastest First Capture: {report['comparison']['winner_by_time_to_first']}")
    
    # Generate plot
    print("\nGenerating comparison plot...")
    plot_cumulative_captures(results1, results2, results3)


def run_cohesion_analysis(predator_counts: list = None, duration: int = 5000):
    """
    Run cohesion time-series analysis only.
    
    Args:
        predator_counts: List of predator counts to test
        duration: Duration in frames
    """
    set_headless()
    
    from .simulation.benchmark import BenchmarkSimulation
    from .analysis.export import export_cohesion_timeseries_to_csv
    from .analysis.plotting import plot_cohesion_comparison
    from .core.config import SimulationConfig
    
    if predator_counts is None:
        predator_counts = [3, 5, 7, 9]
    
    print("=" * 70)
    print("COHESION TIME-SERIES ANALYSIS")
    print("=" * 70)
    print("Models being compared:")
    print("  - Model 1: Herding (non-intercept)")
    print("  - Model 2: Herding (with intercept)")
    print("  - Model 3: Full Coordination")
    print(f"\nPredator counts: {predator_counts}")
    print(f"Duration: {duration} frames")
    print()
    
    base_config = SimulationConfig().to_dict()
    base_config["splitMergeEnabled"] = False
    
    all_results = {}
    csv_files = []
    
    for pred_count in predator_counts:
        print(f"\n{'=' * 60}")
        print(f"TESTING {pred_count} PREDATORS")
        print(f"{'=' * 60}")
        
        configs = [
            ("Model 1 (Herding)", {"cooperativeHunting": False, "predatorInterception": False, "enableClustering": False}),
            ("Model 2 (Intercept)", {"cooperativeHunting": False, "predatorInterception": True, "enableClustering": False}),
            ("Model 3 (Coordination)", {"cooperativeHunting": True, "predatorInterception": True, "enableClustering": True}),
        ]
        
        results = {}
        for name, overrides in configs:
            print(f"\nRunning {name}...")
            config = base_config.copy()
            config.update(overrides)
            config["predatorCount"] = pred_count
            
            random.seed(42)
            sim = BenchmarkSimulation(config)
            results[name.split()[1].lower().strip("()")] = sim.run_benchmark(duration)
        
        all_results[pred_count] = {
            "model1": results["herding"],
            "model2": results["intercept"],
            "model3": results["coordination"],
        }
        
        csv_file = export_cohesion_timeseries_to_csv(
            results["herding"], results["intercept"], results["coordination"], pred_count
        )
        csv_files.append(csv_file)
        
        # Print summary
        print(f"\nSummary for {pred_count} predators:")
        for key, r in results.items():
            print(f"  {key}: cohesion={r['avg_cohesion']:.2f}, captures={r['total_captures']}")
    
    print("\n" + "=" * 70)
    print("COHESION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated CSV files: {csv_files}")
    
    # Generate plot
    print("\nGenerating cohesion comparison plot...")
    plot_cohesion_comparison(all_results, predator_counts)
    
    return all_results, csv_files


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Boids Simulation with Cooperative Hunting")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")
    parser.add_argument("--cohesion", action="store_true", help="Run cohesion analysis only")
    parser.add_argument("--trials", type=int, default=50, help="Number of benchmark trials")
    parser.add_argument("--duration", type=int, default=5000, help="Simulation duration in frames")
    parser.add_argument("--record-video", action="store_true", help="Record video during benchmark")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(
            num_trials=args.trials,
            duration=args.duration,
            record_video=args.record_video
        )
    elif args.cohesion:
        run_cohesion_analysis(duration=args.duration)
    else:
        run_interactive()


if __name__ == "__main__":
    main()

