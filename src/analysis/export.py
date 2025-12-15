"""
Export functions for saving benchmark results to CSV and JSON.
"""

import csv
import json
from typing import Dict, List, Any


def export_results_to_csv(results1: List[Dict], results2: List[Dict], 
                          results3: List[Dict], filename: str = "benchmark_results.csv") -> str:
    """
    Export benchmark results to CSV format.
    
    Args:
        results1: Results from method 1 (basic herding)
        results2: Results from method 2 (herding + interception)
        results3: Results from method 3 (full coordination)
        filename: Output filename
        
    Returns:
        Path to saved CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['simulation_id', 'boids_caught', 'avg_stress', 
                     'avg_cohesion', 'avg_predator_speed', 'final_boid_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in results1:
            writer.writerow({
                'simulation_id': f"method1_trial{result['trial']}",
                'boids_caught': result['total_captures'],
                'avg_stress': result['avg_stress'],
                'avg_cohesion': result['avg_cohesion'],
                'avg_predator_speed': result['avg_predator_speed'],
                'final_boid_count': result['final_prey_count']
            })
        
        for result in results2:
            writer.writerow({
                'simulation_id': f"method2_trial{result['trial']}",
                'boids_caught': result['total_captures'],
                'avg_stress': result['avg_stress'],
                'avg_cohesion': result['avg_cohesion'],
                'avg_predator_speed': result['avg_predator_speed'],
                'final_boid_count': result['final_prey_count']
            })
        
        for result in results3:
            writer.writerow({
                'simulation_id': f"method3_trial{result['trial']}",
                'boids_caught': result['total_captures'],
                'avg_stress': result['avg_stress'],
                'avg_cohesion': result['avg_cohesion'],
                'avg_predator_speed': result['avg_predator_speed'],
                'final_boid_count': result['final_prey_count']
            })
    
    print(f"\nCSV results saved to: {filename}")
    return filename


def export_cohesion_timeseries_to_csv(results1: Dict, results2: Dict, results3: Dict,
                                      predator_count: int, 
                                      filename: str = None) -> str:
    """
    Export cohesion time-series data for all 3 models to CSV.
    
    Args:
        results1: Results from model 1 (herding, no intercept)
        results2: Results from model 2 (herding + intercept)
        results3: Results from model 3 (full coordination)
        predator_count: Number of predators used
        filename: Output filename (auto-generated if None)
        
    Returns:
        Path to saved CSV file
    """
    if filename is None:
        filename = f"cohesion_timeseries_{predator_count}_predators.csv"
    
    cohesion1 = results1["cohesion_over_time"]
    cohesion2 = results2["cohesion_over_time"]
    cohesion3 = results3["cohesion_over_time"]
    
    # Create lookup dictionaries
    cohesion1_dict = {e["frame"]: e["cohesion"] for e in cohesion1}
    cohesion2_dict = {e["frame"]: e["cohesion"] for e in cohesion2}
    cohesion3_dict = {e["frame"]: e["cohesion"] for e in cohesion3}
    
    boidcount1_dict = {e["frame"]: e["boid_count"] for e in cohesion1}
    boidcount2_dict = {e["frame"]: e["boid_count"] for e in cohesion2}
    boidcount3_dict = {e["frame"]: e["boid_count"] for e in cohesion3}
    
    all_frames = sorted(set(cohesion1_dict.keys()) | 
                       set(cohesion2_dict.keys()) | 
                       set(cohesion3_dict.keys()))
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['frame', 
                     'model1_herding_cohesion', 'model2_intercept_cohesion', 
                     'model3_coordination_cohesion',
                     'model1_boid_count', 'model2_boid_count', 'model3_boid_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for frame in all_frames:
            writer.writerow({
                'frame': frame,
                'model1_herding_cohesion': f"{cohesion1_dict.get(frame, ''):.2f}" if frame in cohesion1_dict else '',
                'model2_intercept_cohesion': f"{cohesion2_dict.get(frame, ''):.2f}" if frame in cohesion2_dict else '',
                'model3_coordination_cohesion': f"{cohesion3_dict.get(frame, ''):.2f}" if frame in cohesion3_dict else '',
                'model1_boid_count': boidcount1_dict.get(frame, ''),
                'model2_boid_count': boidcount2_dict.get(frame, ''),
                'model3_boid_count': boidcount3_dict.get(frame, ''),
            })
    
    print(f"  Cohesion time-series saved to: {filename}")
    return filename


def export_benchmark_report(results: Dict[str, Any], filename: str = "hunting_benchmark_results.json") -> str:
    """
    Export full benchmark report to JSON.
    
    Args:
        results: Complete benchmark results dictionary
        filename: Output filename
        
    Returns:
        Path to saved JSON file
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark report saved to: {filename}")
    return filename


def calculate_aggregate_stats(trial_results: List[Dict]) -> Dict[str, float]:
    """
    Calculate mean and standard deviation across trials.
    
    Args:
        trial_results: List of result dictionaries from multiple trials
        
    Returns:
        Dictionary with mean and std for each metric
    """
    import math
    
    if not trial_results:
        return {}
    
    metrics = [
        "total_captures", "total_distance_traveled", "captures_per_distance",
        "captures_per_frame", "burst_success_rate", "avg_formation_quality",
        "avg_stress", "avg_cohesion", "avg_predator_speed", "first_capture_frame", 
        "avg_time_between_captures", "burst_captures", "herding_captures", 
        "elapsed_time_seconds"
    ]
    
    aggregates = {}
    
    for metric in metrics:
        values = [r[metric] for r in trial_results if metric in r and r[metric] is not None]
        if values:
            mean = sum(values) / len(values)
            aggregates[f"{metric}_mean"] = mean
            if len(values) > 1:
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                aggregates[f"{metric}_std"] = math.sqrt(variance)
            else:
                aggregates[f"{metric}_std"] = 0
    
    return aggregates

