import json
import numpy as np
from pathlib import Path

def load_results(filename="hunting_benchmark_results.json"):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_method_stats(method_name, data):
    """Print detailed statistics for a single method"""
    agg = data.get("aggregates", {})
    config = data.get("config", {})
    
    print(f"\n{method_name}")
    print("-" * 80)
    
    # Configuration
    print("\nConfiguration:")
    print(f"  Cooperative Hunting: {config.get('cooperativeHunting', 'N/A')}")
    print(f"  Interception:        {config.get('predatorInterception', 'N/A')}")
    print(f"  Clustering:          {config.get('enableClustering', 'N/A')}")
    
    # Performance Metrics
    print("\nPerformance Metrics:")
    print(f"  Total Captures:      {agg.get('total_captures_mean', 0):.2f} ± {agg.get('total_captures_std', 0):.2f}")
    print(f"  Captures/Frame:      {agg.get('captures_per_frame_mean', 0):.6f} ± {agg.get('captures_per_frame_std', 0):.6f}")
    print(f"  Captures/Distance:   {agg.get('captures_per_distance_mean', 0):.6f} ± {agg.get('captures_per_distance_std', 0):.6f}")
    
    # Timing Metrics
    print("\nTiming Metrics:")
    print(f"  First Capture:       {agg.get('first_capture_frame_mean', 0):.0f} ± {agg.get('first_capture_frame_std', 0):.0f} frames")
    print(f"  Avg Time Between:    {agg.get('avg_time_between_captures_mean', 0):.1f} ± {agg.get('avg_time_between_captures_std', 0):.1f} frames")
    print(f"  Execution Time:      {agg.get('elapsed_time_seconds_mean', 0):.1f} ± {agg.get('elapsed_time_seconds_std', 0):.1f} seconds")
    
    # Movement Metrics
    print("\nMovement Metrics:")
    print(f"  Total Distance:      {agg.get('total_distance_traveled_mean', 0):.0f} ± {agg.get('total_distance_traveled_std', 0):.0f}")
    
    # Capture Distribution
    print("\nCapture Distribution:")
    print(f"  Burst Captures:      {agg.get('burst_captures_mean', 0):.2f} ± {agg.get('burst_captures_std', 0):.2f}")
    print(f"  Herding Captures:    {agg.get('herding_captures_mean', 0):.2f} ± {agg.get('herding_captures_std', 0):.2f}")
    
    # Coordination-specific metrics
    if config.get('cooperativeHunting'):
        print("\nCoordination Metrics:")
        print(f"  Formation Quality:   {agg.get('avg_formation_quality_mean', 0):.3f} ± {agg.get('avg_formation_quality_std', 0):.3f}")
        print(f"  Burst Success Rate:  {agg.get('burst_success_rate_mean', 0):.2%} ± {agg.get('burst_success_rate_std', 0):.2%}")
    
    # Prey Behavior
    print("\nPrey Response:")
    print(f"  Avg Stress:          {agg.get('avg_stress_mean', 0):.3f} ± {agg.get('avg_stress_std', 0):.3f}")
    print(f"  Avg Cohesion:        {agg.get('avg_cohesion_mean', 0):.1f} ± {agg.get('avg_cohesion_std', 0):.1f}")

def print_comparison_table(results):
    """Print a comparison table of key metrics"""
    print_section_header("QUICK COMPARISON TABLE")
    
    methods = [
        ("Basic Herding", results["method_1_basic_herding"]),
        ("Herding + Interception", results["method_2_herding_interception"]),
        ("Full Coordination", results["method_3_full_coordination"])
    ]
    
    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Metric", "Basic", "Interception", "Coordination"
    ))
    print("-" * 80)
    
    # Total Captures
    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Total Captures",
        methods[0][1]["aggregates"].get("total_captures_mean", 0),
        methods[1][1]["aggregates"].get("total_captures_mean", 0),
        methods[2][1]["aggregates"].get("total_captures_mean", 0)
    ))
    
    # Efficiency
    print("{:<25} {:>15.6f} {:>15.6f} {:>15.6f}".format(
        "Captures/Frame",
        methods[0][1]["aggregates"].get("captures_per_frame_mean", 0),
        methods[1][1]["aggregates"].get("captures_per_frame_mean", 0),
        methods[2][1]["aggregates"].get("captures_per_frame_mean", 0)
    ))
    
    print("{:<25} {:>15.6f} {:>15.6f} {:>15.6f}".format(
        "Captures/Distance",
        methods[0][1]["aggregates"].get("captures_per_distance_mean", 0),
        methods[1][1]["aggregates"].get("captures_per_distance_mean", 0),
        methods[2][1]["aggregates"].get("captures_per_distance_mean", 0)
    ))
    
    # First Capture
    print("{:<25} {:>15.0f} {:>15.0f} {:>15.0f}".format(
        "First Capture (frames)",
        methods[0][1]["aggregates"].get("first_capture_frame_mean", 0),
        methods[1][1]["aggregates"].get("first_capture_frame_mean", 0),
        methods[2][1]["aggregates"].get("first_capture_frame_mean", 0)
    ))
    
    # Avg Time Between Captures
    print("{:<25} {:>15.1f} {:>15.1f} {:>15.1f}".format(
        "Avg Time Between (frames)",
        methods[0][1]["aggregates"].get("avg_time_between_captures_mean", 0),
        methods[1][1]["aggregates"].get("avg_time_between_captures_mean", 0),
        methods[2][1]["aggregates"].get("avg_time_between_captures_mean", 0)
    ))
    
    # Distance Traveled
    print("{:<25} {:>15.0f} {:>15.0f} {:>15.0f}".format(
        "Total Distance",
        methods[0][1]["aggregates"].get("total_distance_traveled_mean", 0),
        methods[1][1]["aggregates"].get("total_distance_traveled_mean", 0),
        methods[2][1]["aggregates"].get("total_distance_traveled_mean", 0)
    ))
    
    # Prey Stress
    print("{:<25} {:>15.3f} {:>15.3f} {:>15.3f}".format(
        "Prey Stress Level",
        methods[0][1]["aggregates"].get("avg_stress_mean", 0),
        methods[1][1]["aggregates"].get("avg_stress_mean", 0),
        methods[2][1]["aggregates"].get("avg_stress_mean", 0)
    ))

def print_winners(results):
    """Print the winners in each category"""
    print_section_header("WINNERS BY CATEGORY")
    
    comparison = results.get("comparison", {})
    
    print(f"\n  🏆 Most Total Captures:        {comparison.get('winner_by_total_captures', 'N/A')}")
    print(f"  🏆 Best Efficiency:            {comparison.get('winner_by_efficiency', 'N/A')}")
    print(f"  🏆 Fastest First Capture:      {comparison.get('winner_by_time_to_first', 'N/A')}")

def calculate_percentile_stats(trials_data):
    """Calculate min, max, median, and percentiles for trial data"""
    if not trials_data:
        return {}
    
    captures = [t.get("total_captures", 0) for t in trials_data]
    
    return {
        "min": min(captures),
        "max": max(captures),
        "median": np.median(captures),
        "25th_percentile": np.percentile(captures, 25),
        "75th_percentile": np.percentile(captures, 75)
    }

def print_trial_variability(results):
    """Print variability statistics across trials"""
    print_section_header("TRIAL VARIABILITY ANALYSIS")
    
    methods = [
        ("Basic Herding", results["method_1_basic_herding"]),
        ("Herding + Interception", results["method_2_herding_interception"]),
        ("Full Coordination", results["method_3_full_coordination"])
    ]
    
    for name, data in methods:
        trials = data.get("trial_results", [])
        stats = calculate_percentile_stats(trials)
        mean = data["aggregates"].get("total_captures_mean", 0)
        std = data["aggregates"].get("total_captures_std", 0)
        cv = (std / mean * 100) if mean > 0 else 0  # Coefficient of variation
        
        print(f"\n{name}:")
        print(f"  Range:  {stats.get('min', 0):.0f} - {stats.get('max', 0):.0f} captures")
        print(f"  Median: {stats.get('median', 0):.0f} captures")
        print(f"  IQR:    {stats.get('25th_percentile', 0):.0f} - {stats.get('75th_percentile', 0):.0f}")
        print(f"  CV:     {cv:.1f}% (coefficient of variation)")

def print_efficiency_analysis(results):
    """Print efficiency analysis"""
    print_section_header("EFFICIENCY ANALYSIS")
    
    methods = [
        ("Basic Herding", results["method_1_basic_herding"]),
        ("Herding + Interception", results["method_2_herding_interception"]),
        ("Full Coordination", results["method_3_full_coordination"])
    ]
    
    print("\n{:<25} {:>15} {:>20}".format("Method", "Captures/km", "Energy Cost"))
    print("-" * 80)
    
    for name, data in methods:
        agg = data["aggregates"]
        cap_per_dist = agg.get("captures_per_distance_mean", 0)
        total_dist = agg.get("total_distance_traveled_mean", 0)
        total_cap = agg.get("total_captures_mean", 0)
        
        # Convert to captures per 1000 units (km equivalent)
        cap_per_km = cap_per_dist * 1000
        
        # Energy cost = distance per capture
        energy_cost = total_dist / total_cap if total_cap > 0 else 0
        
        print("{:<25} {:>15.3f} {:>20.0f}".format(
            name, cap_per_km, energy_cost
        ))

def print_summary_insights(results):
    """Print key insights from the analysis"""
    print_section_header("KEY INSIGHTS")
    
    methods = [
        ("Basic Herding", results["method_1_basic_herding"]),
        ("Herding + Interception", results["method_2_herding_interception"]),
        ("Full Coordination", results["method_3_full_coordination"])
    ]
    
    # Get total captures for each method
    captures = [(name, data["aggregates"].get("total_captures_mean", 0)) for name, data in methods]
    captures.sort(key=lambda x: x[1], reverse=True)
    
    best_method = captures[0][0]
    worst_method = captures[2][0]
    performance_gap = ((captures[0][1] - captures[2][1]) / captures[2][1] * 100)
    
    print(f"\n1. PERFORMANCE RANKING:")
    for i, (name, cap) in enumerate(captures, 1):
        print(f"   {i}. {name}: {cap:.2f} captures")
    
    print(f"\n2. PERFORMANCE GAP:")
    print(f"   {best_method} outperforms {worst_method} by {performance_gap:.1f}%")
    
    # Time to first capture
    first_captures = [(name, data["aggregates"].get("first_capture_frame_mean", 0)) for name, data in methods]
    first_captures.sort(key=lambda x: x[1])
    
    print(f"\n3. REACTION TIME:")
    print(f"   Fastest to first capture: {first_captures[0][0]} ({first_captures[0][1]:.0f} frames)")
    print(f"   Slowest to first capture: {first_captures[2][0]} ({first_captures[2][1]:.0f} frames)")
    
    # Coordination efficiency
    coord_data = results["method_3_full_coordination"]["aggregates"]
    if coord_data.get("burst_success_rate_mean"):
        print(f"\n4. COORDINATION METRICS:")
        print(f"   Burst Success Rate: {coord_data.get('burst_success_rate_mean', 0):.1%}")
        print(f"   Formation Quality: {coord_data.get('avg_formation_quality_mean', 0):.3f}")
        
        # Analyze if coordination overhead is worth it
        coord_captures = methods[2][1]["aggregates"].get("total_captures_mean", 0)
        simple_captures = methods[0][1]["aggregates"].get("total_captures_mean", 0)
        
        if coord_captures < simple_captures:
            print(f"\n   ⚠️  Coordination overhead exceeds benefits in short simulations")
            print(f"   ⚠️  Consider: longer duration, lower formation threshold, or smaller formation radius")
    
    # Prey stress comparison
    stress_levels = [(name, data["aggregates"].get("avg_stress_mean", 0)) for name, data in methods]
    stress_levels.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n5. PREY STRESS:")
    print(f"   Most stressful to prey: {stress_levels[0][0]} ({stress_levels[0][1]:.3f})")
    print(f"   Least stressful to prey: {stress_levels[2][0]} ({stress_levels[2][1]:.3f})")

def main():
    # Load results
    results_file = "hunting_benchmark_results.json"
    
    if not Path(results_file).exists():
        print(f"Error: Could not find {results_file}")
        print("Please run the benchmark first: python benchmark_hunting_methods.py")
        return
    
    print("Loading benchmark results...")
    results = load_results(results_file)
    
    # Print benchmark configuration
    config = results.get("benchmark_config", {})
    print_section_header("BENCHMARK CONFIGURATION")
    print(f"\n  Duration:       {config.get('duration_frames', 0)} frames")
    print(f"  Trials/Method:  {config.get('trials_per_method', 0)}")
    print(f"  Total Runs:     {config.get('trials_per_method', 0) * 3}")
    
    # Print winners
    print_winners(results)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print detailed stats for each method
    print_section_header("DETAILED METHOD STATISTICS")
    
    print_method_stats(
        "METHOD 1: BASIC HERDING",
        results["method_1_basic_herding"]
    )
    
    print_method_stats(
        "METHOD 2: HERDING + INTERCEPTION",
        results["method_2_herding_interception"]
    )
    
    print_method_stats(
        "METHOD 3: FULL COORDINATION",
        results["method_3_full_coordination"]
    )
    
    # Print trial variability
    print_trial_variability(results)
    
    # Print efficiency analysis
    print_efficiency_analysis(results)
    
    # Print insights
    print_summary_insights(results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
