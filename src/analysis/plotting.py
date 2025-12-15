"""
Plotting functions for visualizing benchmark results.
"""

from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_cumulative_captures(results1: List[Dict], results2: List[Dict], 
                             results3: List[Dict], output_file: str = "hunting_performance_comparison.png") -> str:
    """
    Plot cumulative captures over time for all three hunting methods.
    
    Args:
        results1: Results from method 1 (basic herding)
        results2: Results from method 2 (herding + interception)
        results3: Results from method 3 (full coordination)
        output_file: Output filename for the plot
        
    Returns:
        Path to saved plot file
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping plot.")
        return ""
    
    plt.figure(figsize=(12, 7))
    
    # Use first trial from each method
    trial1_data = results1[0]["cumulative_captures_over_time"]
    trial2_data = results2[0]["cumulative_captures_over_time"]
    trial3_data = results3[0]["cumulative_captures_over_time"]
    
    frames1 = [d["frame"] for d in trial1_data]
    captures1 = [d["captures"] for d in trial1_data]
    
    frames2 = [d["frame"] for d in trial2_data]
    captures2 = [d["captures"] for d in trial2_data]
    
    frames3 = [d["frame"] for d in trial3_data]
    captures3 = [d["captures"] for d in trial3_data]
    
    plt.plot(frames1, captures1, label='Basic Herding', linewidth=2, color='#FF6B6B')
    plt.plot(frames2, captures2, label='Herding + Interception', linewidth=2, color='#4ECDC4')
    plt.plot(frames3, captures3, label='Full Coordination', linewidth=2, color='#95E1D3')
    
    plt.xlabel('Frame Number', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Fish Captured', fontsize=12, fontweight='bold')
    plt.title('Hunting Performance Comparison: Cumulative Captures Over Time', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add final count annotations
    if captures1:
        plt.text(frames1[-1], captures1[-1], f' {captures1[-1]}', 
                 verticalalignment='center', fontsize=9, color='#FF6B6B')
    if captures2:
        plt.text(frames2[-1], captures2[-1], f' {captures2[-1]}', 
                 verticalalignment='center', fontsize=9, color='#4ECDC4')
    if captures3:
        plt.text(frames3[-1], captures3[-1], f' {captures3[-1]}', 
                 verticalalignment='center', fontsize=9, color='#95E1D3')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()
    return output_file


def plot_cohesion_comparison(all_results: Dict[int, Dict], 
                             predator_counts: List[int],
                             output_file: str = "cohesion_comparison_all_predators.png") -> str:
    """
    Create a multi-panel plot showing cohesion over time for all predator counts.
    
    Args:
        all_results: Dictionary mapping predator count to results dict with model1/2/3 keys
        predator_counts: List of predator counts tested
        output_file: Output filename for the plot
        
    Returns:
        Path to saved plot file
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping plot.")
        return ""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {
        'model1': '#FF6B6B',  # Red - Herding (no intercept)
        'model2': '#FFB347',  # Orange - Herding + Intercept
        'model3': '#4ECDC4'   # Teal - Full Coordination
    }
    
    for idx, predator_count in enumerate(predator_counts[:4]):
        ax = axes[idx]
        
        results1 = all_results[predator_count]["model1"]
        results2 = all_results[predator_count]["model2"]
        results3 = all_results[predator_count]["model3"]
        
        cohesion1 = results1["cohesion_over_time"]
        cohesion2 = results2["cohesion_over_time"]
        cohesion3 = results3["cohesion_over_time"]
        
        frames1 = [d["frame"] for d in cohesion1]
        values1 = [d["cohesion"] for d in cohesion1]
        
        frames2 = [d["frame"] for d in cohesion2]
        values2 = [d["cohesion"] for d in cohesion2]
        
        frames3 = [d["frame"] for d in cohesion3]
        values3 = [d["cohesion"] for d in cohesion3]
        
        ax.plot(frames1, values1, label='Model 1: Herding', 
                linewidth=2, color=colors['model1'], alpha=0.8)
        ax.plot(frames2, values2, label='Model 2: Herding + Intercept', 
                linewidth=2, color=colors['model2'], alpha=0.8)
        ax.plot(frames3, values3, label='Model 3: Coordination', 
                linewidth=2, color=colors['model3'], alpha=0.8)
        
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Cohesion (avg dist to centroid)', fontsize=10)
        ax.set_title(f'{predator_count} Predators', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Annotations
        if values1 and values2 and values3:
            ax.annotate(f'{values1[-1]:.0f}', xy=(frames1[-1], values1[-1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, color=colors['model1'])
            ax.annotate(f'{values2[-1]:.0f}', xy=(frames2[-1], values2[-1]), 
                       xytext=(5, 0), textcoords='offset points', 
                       fontsize=8, color=colors['model2'])
            ax.annotate(f'{values3[-1]:.0f}', xy=(frames3[-1], values3[-1]), 
                       xytext=(5, -5), textcoords='offset points', 
                       fontsize=8, color=colors['model3'])
    
    plt.suptitle('Cohesion Over Time: Bait Ball Formation Analysis\n'
                 '(Lower values = tighter prey grouping)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCohesion comparison plot saved to: {output_file}")
    
    plt.show()
    return output_file

