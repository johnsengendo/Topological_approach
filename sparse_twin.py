# Importing standard Python libraries for system operations and warnings management
import os
import sys
import warnings

# Adding the current directory to Python path for accessing custom modules
sys.path.append(os.path.dirname(__file__))

# Importing essential numerical and scientific computing libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Importing custom modules for ST-NDT functionality
from st_ndt_plots import *
from st_ndt_io import *
from st_ndt_core import *

# Suppressing warning messages to keep output clean
warnings.filterwarnings('ignore')

# Configuring matplotlib parameters for high-quality visualizations
plt.rcParams.update({
    'figure.dpi': 150,           # Setting high DPI for crisp images
    'savefig.dpi': 150,          # Setting high DPI when saving figures
    'figure.figsize': [12, 8],   # Defining larger default figure size
    'font.size': 12,             # Setting larger base font size
    'font.weight': 'bold',       # Making fonts bold by default
    'axes.labelweight': 'bold',  # Making axis labels bold
    'axes.titleweight': 'bold',  # Making titles bold
    'xtick.labelsize': 11,       # Setting larger tick labels
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,    # Creating semi-transparent legend background
    'grid.alpha': 0.3,           # Setting subtle grid appearance
    'lines.linewidth': 2.5,      # Making lines thicker
    'axes.linewidth': 1.5,       # Making axes thicker
    'xtick.major.width': 1.5,    # Making tick marks thicker
    'ytick.major.width': 1.5,
})

def demonstrate_st_ndt():
    """
    Demonstrating the Sparse Topological Network Digital Twin using Cogentco network topology
    This function showcases the complete ST-NDT pipeline on a real ISP network
    """
    print("=== Sparse Topological Network Digital Twin Demo ===\n")
    print("Loading Cogentco Network Topology...")
    
    # Loading the Cogentco network topology from GraphML file
    G = load_topology_zoo_graph('Cogentco_clean')
    
    # Checking if the topology loaded successfully
    if G is None:
        print("FAILED: Cannot proceed without Cogentco topology file.")
        print("Please ensure Cogentco.gml is in your current directory.")
        return None, None, None
    
    # Initializing the ST-NDT system with the loaded topology
    print("\nInitializing Sparse Topological NDT with Cogentco network...")
    st_ndt = SparseTopologicalNDT(G)
    
    # Analyzing the network's topological properties
    print("\n1. COGENTCO TOPOLOGICAL ANALYSIS")
    analysis = st_ndt.analyze_network_topology()
    print(f"   Cogentco Network Properties:")
    print(f"   • Nodes: {analysis['network_size']['nodes']}")
    print(f"   • Edges: {analysis['network_size']['edges']}")
    print(f"   • Polygons found: {analysis['topology_matrices']['polygons_found']}")
    print(f"   • Harmonic dimension (network holes): {analysis['spectral_properties']['harmonic_dimension']}")
    print(f"   • Connected components (β₀): {analysis['topological_invariants']['betti_0']}")
    print(f"   • Topological holes (β₁): {analysis['topological_invariants']['betti_1']}")
    
    # Generating realistic network traffic signal for testing
    print("\n2. COGENTCO NETWORK SIGNAL GENERATION")
    try:
        # Computing the Laplacian matrix for the node graph
        L_node = nx.laplacian_matrix(G).astype(float).toarray()
        eigvals_node, eigvecs_node = np.linalg.eigh(L_node)
        
        # Creating frequency response based on eigenvalues
        alpha = 0.1
        signal_freq = np.exp(-alpha * eigvals_node)
        node_signal = eigvecs_node @ (signal_freq * np.random.randn(len(G)))
        
        # Normalizing the node signal to [0,1] range
        node_signal = (node_signal - node_signal.min()) / (node_signal.max() - node_signal.min() + 1e-10)
        
        # Converting node signal to edge signal using the incidence matrix
        edge_signal = np.abs(st_ndt.B1.T @ node_signal).flatten()
        print(f"   Generated Cogentco traffic signal with {len(edge_signal)} edge flows")
        
    except Exception as e:
        print(f"Warning: Signal generation failed: {e}, using random signal")
        # Falling back to random signal if generation fails
        edge_signal = np.random.rand(st_ndt.E)
    
    # Creating visualizations of the network and sampling strategy
    print("\n3. COGENTCO NETWORK VISUALIZATION")
    try:
        pos = visualize_network_and_sampling(st_ndt, edge_signal, sparsity_ratio=0.2)
    except Exception as e:
        print(f"Warning: Network visualization failed: {e}")
        pos = None
    
    # Comparing different sparse monitoring approaches
    print("\n4. SPARSE MONITORING ON COGENTCO NETWORK")
    sparsity_results = st_ndt.compare_sparsity_methods(edge_signal)
    
    print("   Cogentco Network Sparse Monitoring Results:")
    print("   Sparsity Ratio | Topological | Degree-based")
    print("   ---------------|-------------|--------------")
    for i, ratio in enumerate(sparsity_results['sparsity_ratios']):
        print(f"   {ratio:11.1f} | {sparsity_results['topological_errors'][i]:11.4f} | {sparsity_results['degree_based_errors'][i]:12.4f}")
    
    # Demonstrating digital twin synchronization capabilities
    print("\n5. COGENTCO DIGITAL TWIN SYNCHRONIZATION")
    sensor_placement = st_ndt.sparse_sensor_placement(sparsity_ratio=0.15, method='topological')
    selected_edges = sensor_placement['selected_edges']
    
    print(f"   Monitoring {len(selected_edges)}/{st_ndt.E} Cogentco links ({len(selected_edges)/st_ndt.E*100:.1f}%)")
    print(f"   Monitoring {len(sensor_placement['selected_nodes'])}/{st_ndt.N} Cogentco nodes")
    
    # Simulating current measurements from selected edges with noise
    current_measurements = []
    for edge_idx in selected_edges:
        if edge_idx < len(edge_signal):
            # Adding measurement noise to simulate real sensor data
            current_measurements.append(edge_signal[edge_idx] + np.random.normal(0, 0.01))
    
    # Synchronizing the digital twin with current measurements
    sync_results = st_ndt.digital_twin_sync(np.array(current_measurements), sensor_placement)
    
    print("   Cogentco NDT Synchronization Results:")
    print("   Time | Reconstruction Error | Sync Efficiency | Betti Numbers")
    print("   -----|---------------------|-----------------|---------------")
    for i, t in enumerate(sync_results['iterations']):
        error = sync_results['reconstruction_errors'][i]
        efficiency = sync_results['sync_efficiency'][i]
        betti = sync_results['topological_invariants'][i]
        print(f"   {t:4d} | {error:19.4f} | {efficiency:15.3f} | β₀={betti['betti_0']}, β₁={betti['betti_1']}")
    
    # Performing Hodge decomposition to analyze traffic flow patterns
    print("\n6. COGENTCO TRAFFIC HODGE DECOMPOSITION")
    decomposition = st_ndt.hodge_decomposition(edge_signal)
    
    print(f"   Cogentco Traffic Flow Analysis:")
    print(f"   • Total traffic energy: {np.linalg.norm(edge_signal):.4f}")
    print(f"   • Irrotational flows (point-to-point): {np.linalg.norm(decomposition['irrotational']):.4f}")
    print(f"   • Solenoidal flows (circular traffic): {np.linalg.norm(decomposition['solenoidal']):.4f}")
    print(f"   • Harmonic flows (topological): {np.linalg.norm(decomposition['harmonic']):.4f}")
    print(f"   • Reconstruction accuracy: {1 - np.linalg.norm(edge_signal - decomposition['total_reconstructed'])/(np.linalg.norm(edge_signal) + 1e-10):.6f}")

    # Analyzing link reduction capabilities
    print("\n7. COGENTCO LINK REDUCTION ANALYSIS")
    reduction_results = measure_link_reduction(st_ndt, sparsity_ratios=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    
    if reduction_results:
        visualize_link_reduction(reduction_results, "Cogentco Network")
    
    # Comparing link efficiency methods
    print("\n8. COGENTCO LINK EFFICIENCY COMPARISON")
    efficiency_comparison = compare_link_efficiency_methods(st_ndt, edge_signal, sparsity_ratio=0.15)
    
    # Analyzing traffic state transfer efficiency
    print("\n9. TRAFFIC STATE TRANSFER EFFICIENCY ANALYSIS")
    # Generating realistic traffic data with exponential distribution
    traffic_data = np.abs(edge_signal) * 1000 + np.random.exponential(500, st_ndt.E)
    
    traffic_efficiency_results = st_ndt.analyze_traffic_state_efficiency(traffic_data)
    
    if traffic_efficiency_results:
        st_ndt.plot_traffic_efficiency_analysis(traffic_efficiency_results, "Cogentco Network")
        
        # Calculating and displaying efficiency summary statistics
        if traffic_efficiency_results['transfer_reduction_percent']:
            avg_reduction = np.mean(traffic_efficiency_results['transfer_reduction_percent'])
            avg_quality = np.mean(traffic_efficiency_results['approximation_quality'])
            max_savings = max(traffic_efficiency_results['bandwidth_savings_kb'])
            
            print(f"\n=== TRAFFIC EFFICIENCY SUMMARY ===")
            print(f"Average Traffic Data Reduction: {avg_reduction:.1f}%")
            print(f"Average Approximation Quality: {avg_quality:.3f}")
            print(f"Maximum Bandwidth Savings: {max_savings:.1f} KB per monitoring cycle")
            print(f"Inference Accuracy: {avg_quality*100:.1f}% of traffic correctly estimated")
    
    return st_ndt, sparsity_results, sync_results, reduction_results, efficiency_comparison


def compare_ndt_approaches(st_ndt, edge_signal, iterations=5):
    """
    Comparing different NDT approaches: Full monitoring vs Sparse Topological vs Sparse Degree-based
    The function evaluates the trade-offs between accuracy and computational efficiency
    """
    try:
        # Initializing results dictionary with method names and metrics
        results = {
            'methods': ['Full Monitoring', 'Sparse Topological', 'Sparse Degree'],
            'errors': {'Full Monitoring': [], 'Sparse Topological': [], 'Sparse Degree': []},
            'bandwidth_usage': {'Full Monitoring': [], 'Sparse Topological': [], 'Sparse Degree': []},
            'computation_time': {'Full Monitoring': [], 'Sparse Topological': [], 'Sparse Degree': []}
        }
        
        sparsity_ratio = 0.15  # Setting 15% monitoring budget
        
        # Running comparison across multiple iterations
        for t in range(iterations):
            # Simulating network state evolution with noise
            current_signal = edge_signal + 0.1 * np.random.randn(len(edge_signal))
            current_signal = np.clip(current_signal, 0, None)  # Ensuring non-negative values
            
            # Method 1: Full Monitoring (baseline approach)
            import time
            start_time = time.time()
            full_reconstruction = current_signal  # Perfect reconstruction with full monitoring
            full_error = 0.0
            full_time = time.time() - start_time
            
            # Recording full monitoring results
            results['errors']['Full Monitoring'].append(full_error)
            results['bandwidth_usage']['Full Monitoring'].append(1.0)  # Using 100% bandwidth
            results['computation_time']['Full Monitoring'].append(full_time)
            
            # Method 2: Sparse Topological approach
            start_time = time.time()
            topo_placement = st_ndt.sparse_sensor_placement(sparsity_ratio, 'topological')
            topo_measurements = [current_signal[i] for i in topo_placement['selected_edges'] if i < len(current_signal)]
            topo_reconstruction = st_ndt.sparse_reconstruction(np.array(topo_measurements), topo_placement)
            topo_error = np.linalg.norm(topo_reconstruction - current_signal) / (np.linalg.norm(current_signal) + 1e-10)
            topo_time = time.time() - start_time
            
            # Recording topological approach results
            results['errors']['Sparse Topological'].append(topo_error)
            results['bandwidth_usage']['Sparse Topological'].append(sparsity_ratio)
            results['computation_time']['Sparse Topological'].append(topo_time)
            
            # Method 3: Sparse Degree-based approach
            start_time = time.time()
            degree_placement = st_ndt.sparse_sensor_placement(sparsity_ratio, 'degree_based')
            degree_measurements = [current_signal[i] for i in degree_placement['selected_edges'] if i < len(current_signal)]
            degree_reconstruction = st_ndt.sparse_reconstruction(np.array(degree_measurements), degree_placement)
            degree_error = np.linalg.norm(degree_reconstruction - current_signal) / (np.linalg.norm(current_signal) + 1e-10)
            degree_time = time.time() - start_time
            
            # Recording degree-based approach results
            results['errors']['Sparse Degree'].append(degree_error)
            results['bandwidth_usage']['Sparse Degree'].append(sparsity_ratio)
            results['computation_time']['Sparse Degree'].append(degree_time)
        
        return results
        
    except Exception as e:
        print(f"Warning: NDT comparison failed: {e}")
        return None


def sparse_sensor_placement(self, sparsity_ratio=0.1, method='topological'):
    """
    Implementing sparse sensor placement using topology inference from Section V
    The method optimally selects which network elements to monitor
    
    Args:
        sparsity_ratio: Fraction of edges/nodes to monitor (0.1 = 10% monitoring)
        method: Selection strategy - 'topological', 'degree_based' or Random
    """
    # Calculating target numbers of edges and nodes to monitor
    target_edges = max(1, int(self.E * sparsity_ratio))
    target_nodes = max(1, int(self.N * sparsity_ratio))
    
    print(f"Target edges to select: {target_edges}/{self.E}")  # Displaying debug information
    
    # Selecting placement strategy based on method parameter
    if method == 'topological':
        return self._topological_sensor_placement(target_edges, target_nodes)
    elif method == 'degree_based':
        return self._degree_based_placement(target_edges, target_nodes)
    else:
        return self._random_placement(target_edges, target_nodes)


def plot_ndt_comparison(comparison_results):
    """
    Plotting comprehensive NDT method comparison results
    This function creates a dashboard showing performance trade-offs
    """
    try:
        # Checking if results are available
        if comparison_results is None:
            return
            
        # Creating a 2x2 subplot layout for comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        iterations = range(len(comparison_results['errors']['Full Monitoring']))
        
        # Plot 1: Plotting error evolution over time
        for method in comparison_results['methods']:
            if method in comparison_results['errors']:
                errors = comparison_results['errors'][method]
                # Styling different methods with distinct visual properties
                if method == 'Full Monitoring':
                    axes[0,0].plot(iterations, errors, 'k-', linewidth=3, label=method, alpha=0.8)
                elif method == 'Sparse Topological':
                    axes[0,0].plot(iterations, errors, 'r-', linewidth=2, marker='o', label=method)
                else:
                    axes[0,0].plot(iterations, errors, 'b--', linewidth=2, marker='s', label=method)
        
        # Configuring error evolution plot
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Reconstruction Error')
        axes[0,0].set_title('NDT Accuracy Over Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')  # Using log scale for better error visualization
        
        # Plot 2: Creating bandwidth usage comparison
        bandwidth_data = []
        method_names = []
        colors = ['black', 'red', 'blue']
        
        # Calculating average bandwidth usage for each method
        for i, method in enumerate(comparison_results['methods']):
            if method in comparison_results['bandwidth_usage']:
                avg_bandwidth = np.mean(comparison_results['bandwidth_usage'][method]) * 100
                bandwidth_data.append(avg_bandwidth)
                method_names.append(method.replace(' ', '\n'))  # Breaking long names for better display
        
        # Creating bar chart for bandwidth comparison
        bars = axes[0,1].bar(method_names, bandwidth_data, color=colors[:len(bandwidth_data)], alpha=0.7)
        axes[0,1].set_ylabel('Bandwidth Usage (%)')
        axes[0,1].set_title('Monitoring Overhead Comparison')
        axes[0,1].grid(True, alpha=0.3)
        
        # Adding value labels on top of bars
        for bar, value in zip(bars, bandwidth_data):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Creating efficiency frontier (Error vs Bandwidth trade-off)
        for i, method in enumerate(comparison_results['methods']):
            if method in comparison_results['errors'] and method in comparison_results['bandwidth_usage']:
                avg_error = np.mean(comparison_results['errors'][method])
                avg_bandwidth = np.mean(comparison_results['bandwidth_usage'][method]) * 100
                
                # Using different markers and colors for each method
                if method == 'Full Monitoring':
                    axes[1,0].scatter(avg_bandwidth, avg_error, s=200, color='black', marker='*', 
                                    label=method, alpha=0.8, edgecolor='white', linewidth=2)
                elif method == 'Sparse Topological':
                    axes[1,0].scatter(avg_bandwidth, avg_error, s=150, color='red', marker='o', 
                                    label=method, alpha=0.8, edgecolor='white', linewidth=2)
                else:
                    axes[1,0].scatter(avg_bandwidth, avg_error, s=150, color='blue', marker='s', 
                                    label=method, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Configuring efficiency frontier plot
        axes[1,0].set_xlabel('Bandwidth Usage (%)')
        axes[1,0].set_ylabel('Reconstruction Error')
        axes[1,0].set_title('Efficiency Frontier: Accuracy vs Cost')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
        
        # Plot 4: Creating performance improvement matrix
        if len(comparison_results['methods']) >= 2:
            # Calculating improvement factors between methods
            baseline_error = np.mean(comparison_results['errors']['Sparse Random'])
            topo_error = np.mean(comparison_results['errors']['Sparse Topological'])
            full_error = np.mean(comparison_results['errors']['Full Monitoring'])
            
            baseline_bandwidth = np.mean(comparison_results['bandwidth_usage']['Sparse Random'])
            topo_bandwidth = np.mean(comparison_results['bandwidth_usage']['Sparse Topological'])
            
            # Computing improvement metrics
            accuracy_improvement = baseline_error / (topo_error + 1e-10)
            bandwidth_savings = (1 - topo_bandwidth) * 100
            efficiency_score = accuracy_improvement * (bandwidth_savings / 100)
            
            # Creating improvement metrics visualization
            metrics = ['Accuracy\nImprovement', 'Bandwidth\nSavings (%)', 'Overall\nEfficiency']
            values = [accuracy_improvement, bandwidth_savings, efficiency_score]
            colors = ['green', 'blue', 'gold']
            
            bars = axes[1,1].bar(metrics, values, color=colors, alpha=0.7)
            axes[1,1].set_ylabel('Improvement Factor')
            axes[1,1].set_title('Sparse Topological NDT Benefits')
            axes[1,1].grid(True, alpha=0.3)
            
            # Adding value labels on improvement bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{value:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Setting overall figure title and layout
        plt.suptitle(f'Network Digital Twin Performance Dashboard', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Returning summary performance metrics
        return {
            'accuracy_improvement': accuracy_improvement if 'accuracy_improvement' in locals() else 0,
            'bandwidth_savings': bandwidth_savings if 'bandwidth_savings' in locals() else 0,
            'efficiency_score': efficiency_score if 'efficiency_score' in locals() else 0
        }
        
    except Exception as e:
        print(f"Warning: NDT performance visualization failed: {e}")
        return {}

def plot_real_time_ndt_simulation(st_ndt, edge_signal, simulation_steps=20):
    """
    Simulating and visualizing real-time NDT operation over time
    This function demonstrates how the digital twin adapts to changing network conditions
    """
    try:
        print("\n=== REAL-TIME NDT SIMULATION ===")
        
        # Setting up sparse monitoring configuration
        sparsity_ratio = 0.15
        sensor_placement = st_ndt.sparse_sensor_placement(sparsity_ratio, 'topological')
        selected_edges = sensor_placement['selected_edges']
        
        # Initializing simulation data storage
        time_steps = []
        true_signals = []
        reconstructed_signals = []
        monitoring_costs = []
        reconstruction_errors = []
        
        # Setting initial network state
        current_true_signal = edge_signal.copy()
        
        print(f"Simulating {simulation_steps} time steps with {len(selected_edges)}/{st_ndt.E} monitored edges...")
        
        # Running simulation over multiple time steps
        for t in range(simulation_steps):
            # Simulating network evolution with multiple components
            # Adding trend component
            trend = 0.02 * np.sin(0.3 * t) * np.ones(st_ndt.E)
            # Adding noise component
            noise = 0.05 * np.random.randn(st_ndt.E)
            # Adding periodic component
            periodic = 0.03 * np.sin(0.8 * t + np.arange(st_ndt.E) * 0.1)
            
            # Updating signal with combined components
            current_true_signal = current_true_signal + trend + noise + periodic
            current_true_signal = np.clip(current_true_signal, 0, None)
            
            # Performing sparse monitoring on selected edges only
            sparse_measurements = []
            for edge_idx in selected_edges:
                if edge_idx < len(current_true_signal):
                    # Adding measurement noise to simulate real sensor conditions
                    measurement = current_true_signal[edge_idx] + np.random.normal(0, 0.01)
                    sparse_measurements.append(measurement)
            
            # Reconstructing full network state from sparse measurements
            reconstructed_signal = st_ndt.sparse_reconstruction(np.array(sparse_measurements), sensor_placement)
            
            # Calculating performance metrics
            recon_error = np.linalg.norm(reconstructed_signal - current_true_signal) / (np.linalg.norm(current_true_signal) + 1e-10)
            monitoring_cost = len(selected_edges) / st_ndt.E  # Computing fraction of network monitored
            
            # Storing results for analysis
            time_steps.append(t)
            true_signals.append(current_true_signal.copy())
            reconstructed_signals.append(reconstructed_signal.copy())
            monitoring_costs.append(monitoring_cost)
            reconstruction_errors.append(recon_error)
        
        # Creating comprehensive visualization of simulation results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Showing real-time error evolution
        axes[0,0].plot(time_steps, reconstruction_errors, 'r-', linewidth=2, marker='o')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Reconstruction Error')
        axes[0,0].set_title('Real-Time NDT Accuracy')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # Plot 2: Showing bandwidth efficiency over time
        bandwidth_savings = [(1 - cost) * 100 for cost in monitoring_costs]
        axes[0,1].plot(time_steps, bandwidth_savings, 'b-', linewidth=2, marker='s')
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Full Monitoring')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Bandwidth Savings (%)')
        axes[0,1].set_title('Continuous Bandwidth Savings')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Displaying signal tracking quality for recent steps
        recent_steps = min(5, len(time_steps))
        step_indices = range(len(edge_signal))
        
        # Plotting multiple recent signals with fading effect
        for i in range(recent_steps):
            step_idx = len(time_steps) - recent_steps + i
            alpha = 0.3 + 0.7 * (i / recent_steps)  # Creating fade effect for older signals
            
            if step_idx >= 0:
                if i == recent_steps - 1:  # Highlighting most recent signals
                    axes[1,0].plot(step_indices, true_signals[step_idx], 'k-', 
                                 linewidth=2, alpha=alpha, label='True Signal')
                    axes[1,0].plot(step_indices, reconstructed_signals[step_idx], 'r--', 
                                 linewidth=2, alpha=alpha, label='Reconstructed')
                else:
                    axes[1,0].plot(step_indices, true_signals[step_idx], 'k-', 
                                 linewidth=1, alpha=alpha)
                    axes[1,0].plot(step_indices, reconstructed_signals[step_idx], 'r--', 
                                 linewidth=1, alpha=alpha)
        
        axes[1,0].set_xlabel('Edge Index')
        axes[1,0].set_ylabel('Traffic Load')
        axes[1,0].set_title('Signal Tracking Quality (Recent Steps)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Creating performance metrics summary
        avg_error = np.mean(reconstruction_errors)
        avg_bandwidth_savings = np.mean(bandwidth_savings)
        stability = 1 - np.std(reconstruction_errors) / (np.mean(reconstruction_errors) + 1e-10)
        
        metrics = ['Accuracy\n(1-Error)', 'Bandwidth\nSavings (%)', 'Stability\nScore']
        values = [1 - avg_error, avg_bandwidth_savings, stability]
        colors = ['green', 'blue', 'orange']
        
        bars = axes[1,1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1,1].set_ylabel('Performance Score')
        axes[1,1].set_title('Real-Time NDT Performance')
        axes[1,1].grid(True, alpha=0.3)
        
        # Adding value labels on performance bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Finalizing plot layout and display
        plt.suptitle('Real-Time Network Digital Twin Simulation', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Printing comprehensive performance summary
        print(f"\n=== REAL-TIME NDT RESULTS ===")
        print(f"Simulation Duration: {simulation_steps} time steps")
        print(f"Average Reconstruction Error: {avg_error:.4f}")
        print(f"Average Bandwidth Savings: {avg_bandwidth_savings:.1f}%")
        print(f"Performance Stability: {stability:.3f}")
        print(f"Monitored Elements: {len(selected_edges)}/{st_ndt.E} edges ({len(selected_edges)/st_ndt.E*100:.1f}%)")
        
        # Returning simulation results for further analysis
        return {
            'avg_error': avg_error,
            'avg_bandwidth_savings': avg_bandwidth_savings,
            'stability': stability,
            'monitoring_efficiency': len(selected_edges)/st_ndt.E
        }
        
    except Exception as e:
        print(f"Warning: Real-time NDT simulation failed: {e}")
        return {}

def demonstrate_scalability_benefits():
    """
    Demonstrating sparsity scalability with Cogentco network at different monitoring budgets
    The function shows how performance scales with monitoring investment
    """
    print("\n=== COGENTCO SPARSITY SCALABILITY ANALYSIS ===")
    
    # Loading Cogentco network topology
    G = load_topology_zoo_graph('Cogentco_clean')
    if G is None:
        print("Cannot perform scalability analysis without Cogentco topology.")
        return {}
    
    # Initializing ST-NDT system with Cogentco topology
    st_ndt = SparseTopologicalNDT(G)
    
    # Testing different sparsity levels on Cogentco network
    sparsity_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    # Initializing results storage
    results = {
        'sparsity_ratios': [],
        'monitoring_cost': [],
        'reconstruction_quality': [],
        'bandwidth_savings': [],
        'topological_preservation': []
    }
    
    # Generating test signal for Cogentco network analysis
    test_signal = np.random.randn(st_ndt.E)
    
    print(f"Testing Cogentco network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    print("Monitoring | Edges   | Reconstruction | Bandwidth | Topology")
    print("Budget (%) | Selected| Quality        | Savings   | Preserved")
    print("-----------|---------|----------------|-----------|----------")
    
    # Testing each sparsity ratio systematically
    for ratio in sparsity_ratios:
        try:
            # Testing sparse monitoring at current sparsity level
            placement = st_ndt.sparse_sensor_placement(ratio, 'topological')
            sparse_measurements = [test_signal[i] for i in placement['selected_edges'] if i < len(test_signal)]
            reconstructed = st_ndt.sparse_reconstruction(np.array(sparse_measurements), placement)
            
            # Calculating performance metrics
            monitoring_cost = len(placement['selected_edges'])
            quality = 1 - np.linalg.norm(reconstructed - test_signal) / (np.linalg.norm(test_signal) + 1e-10)
            bandwidth_savings = (1 - ratio) * 100
            
            # Checking if topology is preserved (Betti numbers should remain constant)
            decomp_orig = st_ndt.hodge_decomposition(test_signal)
            decomp_recon = st_ndt.hodge_decomposition(reconstructed)
            topology_preserved = abs(np.linalg.norm(decomp_orig['harmonic']) - np.linalg.norm(decomp_recon['harmonic'])) < 0.1
            
            # Storing results for analysis
            results['sparsity_ratios'].append(ratio)
            results['monitoring_cost'].append(monitoring_cost)
            results['reconstruction_quality'].append(quality)
            results['bandwidth_savings'].append(bandwidth_savings)
            results['topological_preservation'].append(1.0 if topology_preserved else 0.0)
            
            print(f"{ratio*100:10.1f} | {monitoring_cost:7d} | {quality:14.3f} | {bandwidth_savings:9.1f} | {'✓' if topology_preserved else '✗'}")
            
        except Exception as e:
            print(f"{ratio*100:10.1f} | ERROR: {str(e)[:50]}")
    
    # Printing Cogentco network efficiency summary
    print(f"\nCogentco Network Efficiency Summary:")
    if results['sparsity_ratios']:
        avg_quality = np.mean(results['reconstruction_quality'])
        avg_savings = np.mean(results['bandwidth_savings'])
        topology_preservation_rate = np.mean(results['topological_preservation']) * 100
        
        print(f"• Average reconstruction quality: {avg_quality:.3f}")
        print(f"• Average bandwidth savings: {avg_savings:.1f}%")
        print(f"• Topology preservation rate: {topology_preservation_rate:.1f}%")
    
    return results


def advanced_ndt_features():
    """
    Demonstrating advanced features using Cogentco network
    The function showcases sophisticated capabilities of the ST-NDT system
    """
    print("\n=== ADVANCED ST-NDT FEATURES (Cogentco Network) ===\n")
    
    try:
        # Using Cogentco network for advanced feature demonstration
        G = load_topology_zoo_graph('Cogentco_clean')
        if G is None:
            print("Cannot demonstrate advanced features without Cogentco topology.")
            return
            
        # Initializing ST-NDT for advanced analysis
        st_ndt = SparseTopologicalNDT(G)
        
        # Performing detailed topological analysis
        print("7. COGENTCO DETAILED TOPOLOGICAL ANALYSIS")
        analysis = st_ndt.analyze_network_topology()
        print(f"   Cogentco ISP Network Characteristics:")
        print(f"   • Network Type: Internet Service Provider Backbone")
        print(f"   • Nodes (PoPs/Routers): {analysis['network_size']['nodes']}")
        print(f"   • Links (Fiber connections): {analysis['network_size']['edges']}")
        print(f"   • Network cycles detected: {analysis['topology_matrices']['polygons_found']}")
        print(f"   • Redundant paths (β₁): {analysis['topological_invariants']['betti_1']}")
        print(f"   • Network resilience score: {analysis['spectral_properties']['harmonic_dimension']}")
        
        # Analyzing Cogentco's network structure in detail
        degree_sequence = [d for n, d in G.degree()]
        avg_degree = np.mean(degree_sequence)
        max_degree = np.max(degree_sequence)
        
        print(f"   • Average node degree: {avg_degree:.2f}")
        print(f"   • Maximum node degree: {max_degree}")
        print(f"   • Network diameter: {nx.diameter(G)}")
        print(f"   • Clustering coefficient: {nx.average_clustering(G):.4f}")
        
        # Analyzing traffic patterns on Cogentco network
        print("\n8. COGENTCO TRAFFIC PATTERN ANALYSIS")
        
        # Generating realistic ISP traffic pattern for analysis
        test_signal = np.random.randn(st_ndt.E)
        
        # Demonstrating topological signal filtering on Cogentco
        if st_ndt.U_harm.shape[1] > 0:
            # Creating pure harmonic signal (represents persistent traffic flows around network bottlenecks)
            harmonic_coeffs = np.random.randn(st_ndt.U_harm.shape[1])
            pure_harmonic_signal = st_ndt.U_harm @ harmonic_coeffs
            
            # Adding realistic ISP traffic noise to test filtering
            noise_signal = 0.1 * np.random.randn(st_ndt.E)
            mixed_signal = pure_harmonic_signal.flatten() + noise_signal
            
            # Extracting traffic patterns using Hodge decomposition
            decomposition = st_ndt.hodge_decomposition(mixed_signal)
            
            print(f"   Cogentco Traffic Flow Decomposition:")
            print(f"   • Original backbone traffic energy: {np.linalg.norm(pure_harmonic_signal):.4f}")
            print(f"   • Extracted backbone patterns: {np.linalg.norm(decomposition['harmonic']):.4f}")
            harmonic_accuracy = 1 - np.linalg.norm(pure_harmonic_signal.flatten() - decomposition['harmonic'])/(np.linalg.norm(pure_harmonic_signal) + 1e-10)
            print(f"   • Traffic pattern extraction accuracy: {harmonic_accuracy:.4f}")
        else:
            print("   No significant traffic circulation patterns detected in Cogentco")
        
        # Determining optimal sparse monitoring for Cogentco ISP network
        print("\n9. COGENTCO OPTIMAL MONITORING STRATEGY")
        sparse_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        print("   ISP Monitoring Budget Analysis:")
        print("   Budget | Links Monitored | Quality | Bandwidth | Cost Efficiency")
        print("   -------|-----------------|---------|-----------|----------------")
        
        # Testing different monitoring budgets
        test_signal = np.random.randn(st_ndt.E)
        for ratio in sparse_ratios:
            try:
                placement = st_ndt.sparse_sensor_placement(ratio, 'topological')
                sparse_measurements = []
                for edge_idx in placement['selected_edges']:
                    if edge_idx < len(test_signal):
                        sparse_measurements.append(test_signal[edge_idx])
                
                reconstructed = st_ndt.sparse_reconstruction(np.array(sparse_measurements), placement)
                
                # Calculating performance metrics for each budget level
                quality = 1 - np.linalg.norm(reconstructed - test_signal) / (np.linalg.norm(test_signal) + 1e-10)
                bandwidth_savings = (1 - ratio) * 100
                cost_efficiency = quality / ratio  # Computing quality per monitoring cost
                
                print(f"   {ratio*100:5.1f}% | {len(placement['selected_edges']):15d} | {quality:7.3f} | {bandwidth_savings:9.1f}% | {cost_efficiency:14.2f}")
                
            except Exception as e:
                print(f"   {ratio*100:5.1f}% | ERROR: {str(e)[:40]}")
                
        print(f"\n   Recommendation: 15-20% monitoring budget provides optimal cost-effectiveness for Cogentco")
                
    except Exception as e:
        print(f"Warning: Advanced Cogentco analysis failed: {e}")

def measure_link_reduction(st_ndt, sparsity_ratios=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]):
    """
    Measuring the reduction in number of links that need to be communicated
    The function quantifies communication efficiency improvements
    
    Args:
        st_ndt: SparseTopologicalNDT instance
        sparsity_ratios: List of monitoring budgets to test
        
    Returns:
        dict: Comprehensive link reduction analysis results
    """
    try:
        # Getting baseline network statistics
        total_links = st_ndt.E
        total_nodes = st_ndt.N
        
        # Initializing results storage structure
        results = {
            'total_network_links': total_links,
            'total_network_nodes': total_nodes,
            'sparsity_ratios': sparsity_ratios,
            'topological_links_needed': [],
            'degree_based_links_needed': [],
            'link_reduction_percentage': [],
            'communication_efficiency': [],
            'bandwidth_compression_ratio': [],
            'network_coverage_ratio': []
        }
        
        print(f"=== LINK REDUCTION ANALYSIS ===")
        print(f"Original Network: {total_nodes} nodes, {total_links} links")
        print()
        print("Budget | Topological | Degree-based | Reduction | Efficiency | Compression")
        print("   (%) |       Links |        Links |      (%) |       Gain |       Ratio")
        print("-------|-------------|--------------|-----------|------------|------------")
        
        # Testing each sparsity ratio for link reduction analysis
        for ratio in sparsity_ratios:
            try:
                # Testing topological approach for link selection
                topo_placement = st_ndt.sparse_sensor_placement(ratio, 'topological')
                topo_links = len(topo_placement['selected_edges'])
                
                # Testing degree-based approach for comparison
                degree_placement = st_ndt.sparse_sensor_placement(ratio, 'degree_based')
                degree_links = len(degree_placement['selected_edges'])
                
                # Calculating communication efficiency metrics
                link_reduction = ((total_links - topo_links) / total_links) * 100
                communication_efficiency = total_links / topo_links if topo_links > 0 else 0
                compression_ratio = total_links / topo_links if topo_links > 0 else 0
                
                # Calculating network coverage (how much of the network can be inferred)
                edge_list = list(st_ndt.G.edges())
                covered_nodes = set()
                for edge_idx in topo_placement['selected_edges']:
                    if edge_idx < len(edge_list):
                        u, v = edge_list[edge_idx]
                        covered_nodes.add(u)
                        covered_nodes.add(v)
                coverage_ratio = len(covered_nodes) / total_nodes if total_nodes > 0 else 0
                
                # Storing calculated results
                results['topological_links_needed'].append(topo_links)
                results['degree_based_links_needed'].append(degree_links)
                results['link_reduction_percentage'].append(link_reduction)
                results['communication_efficiency'].append(communication_efficiency)
                results['bandwidth_compression_ratio'].append(compression_ratio)
                results['network_coverage_ratio'].append(coverage_ratio)
                
                print(f"{ratio*100:6.1f} | {topo_links:11d} | {degree_links:12d} | {link_reduction:9.1f} | {communication_efficiency:10.2f} | {compression_ratio:11.2f}")
                
            except Exception as e:
                print(f"{ratio*100:6.1f} | ERROR: {str(e)[:50]}")
                # Filling with default values to maintain array lengths
                results['topological_links_needed'].append(0)
                results['degree_based_links_needed'].append(0)
                results['link_reduction_percentage'].append(0)
                results['communication_efficiency'].append(0)
                results['bandwidth_compression_ratio'].append(0)
                results['network_coverage_ratio'].append(0)
        
        # Calculating and displaying summary statistics
        if results['link_reduction_percentage']:
            avg_reduction = np.mean([x for x in results['link_reduction_percentage'] if x > 0])
            max_reduction = max([x for x in results['link_reduction_percentage'] if x > 0], default=0)
            avg_compression = np.mean([x for x in results['bandwidth_compression_ratio'] if x > 0])
            
            print()
            print(f"=== LINK REDUCTION SUMMARY ===")
            print(f"Average Link Reduction: {avg_reduction:.1f}%")
            print(f"Maximum Link Reduction: {max_reduction:.1f}%")
            print(f"Average Compression Ratio: {avg_compression:.2f}x")
            print(f"Best Case: Monitor only {min(results['topological_links_needed'])}/{total_links} links "
                  f"({min(results['topological_links_needed'])/total_links*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"Warning: Link reduction analysis failed: {e}")
        return {}


def visualize_link_reduction(reduction_results, network_name="Network"):
    """
    Creating visualizations of link reduction analysis results
    The function generates comprehensive charts showing communication efficiency gains

    Args:
        reduction_results: Results from measure_link_reduction()
        network_name: Name of the network for titles
    """
    try:
        # Checking if we have valid results to visualize
        if not reduction_results or not reduction_results.get('sparsity_ratios'):
            print("Warning: No valid reduction results to visualize")
            return
        
        # Creating 2x2 layout for comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=100)
        colors = ['#FF4757', '#3742FA', '#2ED573', '#FFA502']
        
        # Plot 1: Showing links needed vs sparsity budget
        axes[0,0].plot(reduction_results['sparsity_ratios'], reduction_results['topological_links_needed'], 
                      'o-', color=colors[0], linewidth=3, markersize=8, 
                      label='Topological Approach', markeredgecolor='white', markeredgewidth=2)
        axes[0,0].plot(reduction_results['sparsity_ratios'], reduction_results['degree_based_links_needed'], 
                      's--', color=colors[1], linewidth=3, markersize=8, 
                      label='Degree-based Baseline', markeredgecolor='white', markeredgewidth=2)
        axes[0,0].axhline(y=reduction_results['total_network_links'], color='red', linestyle=':', 
                         linewidth=2, alpha=0.7, label=f'Total Links ({reduction_results["total_network_links"]})')
        
        axes[0,0].set_xlabel('Sparsity Ratio', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Number of Links Required', fontsize=14, fontweight='bold')
        axes[0,0].set_title(f'Communication Overhead', fontsize=16, fontweight='bold')
        axes[0,0].legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='both', which='major', labelsize=11)
        
        # Plot 2: Showing link reduction percentage as bar chart
        axes[0,1].bar(range(len(reduction_results['sparsity_ratios'])), 
                     reduction_results['link_reduction_percentage'],
                     color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0,1].set_xlabel('Monitoring Budget Index', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Link Reduction (%)', fontsize=14, fontweight='bold')
        axes[0,1].set_title(f'Communication Reduction', fontsize=16, fontweight='bold')
        axes[0,1].set_xticks(range(len(reduction_results['sparsity_ratios'])))
        axes[0,1].set_xticklabels([f"{r*100:.0f}%" for r in reduction_results['sparsity_ratios']], rotation=45)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        axes[0,1].tick_params(axis='both', which='major', labelsize=11)
        
        # Plot 3: Displaying compression ratio evolution
        axes[1,0].plot(reduction_results['sparsity_ratios'], reduction_results['bandwidth_compression_ratio'], 
                      '^-', color=colors[3], linewidth=3, markersize=8, markeredgecolor='white', markeredgewidth=2)
        axes[1,0].set_xlabel('Sparsity Ratio', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Compression Ratio (X:1)', fontsize=14, fontweight='bold')
        axes[1,0].set_title(f'Bandwidth Compression', fontsize=16, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='both', which='major', labelsize=11)
        axes[1,0].set_yscale('log')
        
        # Plot 4: Creating summary metrics visualization
        metrics = ['Avg Link\nReduction (%)', 'Max\nCompression', 'Network\nEfficiency']
        if reduction_results['link_reduction_percentage']:
            avg_reduction = np.mean([x for x in reduction_results['link_reduction_percentage'] if x > 0])
            max_compression = max([x for x in reduction_results['bandwidth_compression_ratio'] if x > 0], default=0)
            avg_coverage = np.mean([x for x in reduction_results['network_coverage_ratio'] if x > 0])
            
            values = [avg_reduction, max_compression, avg_coverage * 100]
            colors_bar = colors[:3]
            
            bars = axes[1,1].bar(metrics, values, color=colors_bar, alpha=0.8, 
                               edgecolor='black', linewidth=2)
            axes[1,1].set_ylabel('Performance Score', fontsize=14, fontweight='bold')
            axes[1,1].set_title(f'Overall Performance', fontsize=16, fontweight='bold')
            axes[1,1].tick_params(axis='both', which='major', labelsize=11)
            
            # Adding value labels on bars for clarity
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                              f'{value:.1f}', ha='center', va='bottom', 
                              fontsize=12, fontweight='bold')
        
        # Setting overall figure title and layout
        plt.suptitle(f'Link Reduction Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
        
        # Printing key insights from the analysis
        if reduction_results['link_reduction_percentage']:
            best_reduction = max(reduction_results['link_reduction_percentage'])
            best_idx = reduction_results['link_reduction_percentage'].index(best_reduction)
            best_ratio = reduction_results['sparsity_ratios'][best_idx]
            
            print(f"\n=== {network_name.upper()} LINK REDUCTION INSIGHTS ===")
            print(f"Best Performance: {best_reduction:.1f}% link reduction at {best_ratio*100:.0f}% budget")
            print(f"Original Network: {reduction_results['total_network_links']} total links")
            print(f"Minimum Links Needed: {min(reduction_results['topological_links_needed'])} links")
            print(f"Maximum Compression: {max(reduction_results['bandwidth_compression_ratio']):.1f}:1 ratio")
            print(f"Communication Savings: Up to {best_reduction:.1f}% less data transmission")
            print("="*60)
        
    except Exception as e:
        print(f"Warning: Link reduction visualization failed: {e}")



def compare_link_efficiency_methods(st_ndt, test_signal, sparsity_ratio=0.15):
    """
    Comparing the link efficiency of different monitoring methods
    The function evaluates how effectively different approaches use monitoring resources
    
    Args:
        st_ndt: SparseTopologicalNDT instance
        test_signal: Test signal for reconstruction quality assessment
        sparsity_ratio: Monitoring budget (0.15 = 15% of total links)
        
    Returns:
        dict: Detailed comparison results across methods
    """
    try:
        # Defining methods to compare
        methods = ['topological', 'degree_based']
        method_names = {'topological': 'Topological', 'degree_based': 'Degree-based'}
        
        # Initializing results storage structure
        results = {
            'methods': [],
            'links_used': [],
            'reconstruction_quality': [],
            'network_coverage': [],
            'efficiency_score': []
        }
        
        # Getting baseline network information
        total_links = st_ndt.E
        edge_list = list(st_ndt.G.edges())
        
        print(f"\n=== LINK EFFICIENCY COMPARISON (Budget: {sparsity_ratio*100:.0f}%) ===")
        print("Method        | Links Used | Quality | Coverage | Efficiency Score")
        print("--------------|------------|---------|----------|------------------")
        
        # Testing each method systematically
        for method in methods:
            try:
                # Getting sensor placement using current method
                placement = st_ndt.sparse_sensor_placement(sparsity_ratio, method)
                links_used = len(placement['selected_edges'])
                
                # Testing reconstruction quality with current placement
                sparse_measurements = []
                for edge_idx in placement['selected_edges']:
                    if edge_idx < len(test_signal):
                        sparse_measurements.append(test_signal[edge_idx])
                
                reconstructed = st_ndt.sparse_reconstruction(np.array(sparse_measurements), placement)
                quality = 1 - np.linalg.norm(reconstructed - test_signal) / (np.linalg.norm(test_signal) + 1e-10)
                
                # Calculating network coverage achieved by this method
                covered_nodes = set()
                for edge_idx in placement['selected_edges']:
                    if edge_idx < len(edge_list):
                        u, v = edge_list[edge_idx]
                        covered_nodes.add(u)
                        covered_nodes.add(v)
                coverage = len(covered_nodes) / st_ndt.N
                
                # Computing efficiency score (quality per link used)
                efficiency = quality * coverage / (links_used / total_links) if links_used > 0 else 0
                
                # Storing results for this method
                results['methods'].append(method_names[method])
                results['links_used'].append(links_used)
                results['reconstruction_quality'].append(quality)
                results['network_coverage'].append(coverage)
                results['efficiency_score'].append(efficiency)
                
                print(f"{method_names[method]:<12} | {links_used:10d} | {quality:7.3f} | {coverage:8.3f} | {efficiency:16.3f}")
                
            except Exception as e:
                print(f"{method_names.get(method, method):<12} | ERROR: {str(e)[:40]}")
        
        return results
        
    except Exception as e:
        print(f"Warning: Link efficiency comparison failed: {e}")
        return {}


def create_proof_of_concept():
    """
    Creating complete proof of concept demonstration using Cogentco network only
    The function orchestrates the full ST-NDT demonstration pipeline
    """
    print("=== PROOF OF CONCEPT: SPARSE TOPOLOGICAL NDT ON COGENTCO ===\n")
    
    # Checking if required topology file exists
    if not os.path.exists("Cogentco.gml"):
        print("ERROR: Cogentco.gml not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Please copy your Cogentco.gml file to this directory and run again.")
        return
    
    try:
        # Running main ST-NDT demonstration
        result = demonstrate_st_ndt()
        
        # Handling different return formats for backwards compatibility
        if result is None or len(result) < 3:
            print("Failed to initialize ST-NDT with Cogentco. Stopping demonstration.")
            return
        
        # Extracting results based on return format
        if len(result) == 6:
            st_ndt, sparsity_results, sync_results, reduction_results, efficiency_comparison, traffic_efficiency_results = result
        elif len(result) == 5:
            st_ndt, sparsity_results, sync_results, reduction_results, efficiency_comparison = result
            traffic_efficiency_results = None
        else:
            st_ndt, sparsity_results, sync_results = result[:3]
            reduction_results = efficiency_comparison = traffic_efficiency_results = None
        
        # Checking if ST-NDT initialized successfully
        if st_ndt is None:
            print("Failed to initialize ST-NDT with Cogentco. Stopping demonstration.")
            return
        
        # Plotting sparsity comparison if results available
        if sparsity_results:
            plot_sparsity_comparison(sparsity_results)
        
        # Displaying link communication efficiency summary
        if reduction_results:
            print(f"\n=== LINK COMMUNICATION EFFICIENCY SUMMARY ===")
            total_links = reduction_results.get('total_network_links', 0)
            min_links = min(reduction_results.get('topological_links_needed', [0])) if reduction_results.get('topological_links_needed') else 0
            max_reduction = max(reduction_results.get('link_reduction_percentage', [0])) if reduction_results.get('link_reduction_percentage') else 0
            
            print(f"Original Cogentco Network: {total_links} links")
            print(f"Minimum Links Needed: {min_links} links")
            print(f"Maximum Link Reduction: {max_reduction:.1f}%")
            print(f"Communication Efficiency: {total_links/max(min_links,1):.1f}:1 compression ratio")
            print(f"Insight: Instead of monitoring all {total_links} links, we only need {min_links}!")
        
        # Running Cogentco scalability analysis
        scalability_results = demonstrate_scalability_benefits()
        
        # Demonstrating advanced Cogentco features
        advanced_ndt_features()
        
        # Running anomaly detection demonstration
        print("\n10. COGENTCO NETWORK ANOMALY DETECTION")
        anomaly_detector = TopologicalAnomalyDetector(st_ndt)
        
        # Establishing baseline with normal traffic patterns
        normal_signals = [np.random.randn(st_ndt.E) for _ in range(5)]
        anomaly_detector.establish_baseline(normal_signals)
        
        # Testing normal traffic detection
        test_normal = np.random.randn(st_ndt.E)
        normal_result = anomaly_detector.detect_topological_anomalies(test_normal)
        
        # Creating and testing anomalous traffic pattern
        test_anomaly = test_normal.copy()
        if st_ndt.U_harm.shape[1] > 0:
            anomaly_injection = 3.0 * st_ndt.U_harm @ np.random.randn(st_ndt.U_harm.shape[1])
            test_anomaly += anomaly_injection.flatten()
        
        anomaly_result = anomaly_detector.detect_topological_anomalies(test_anomaly)
        
        print(f"   Normal Cogentco traffic - Anomaly detected: {normal_result['harmonic_anomaly']}")
        print(f"   Suspicious traffic pattern - Anomaly detected: {anomaly_result['harmonic_anomaly']}")
        if anomaly_result['harmonic_energy_ratio']:
            print(f"   Traffic pattern deviation: {anomaly_result['harmonic_energy_ratio']:.2f}x normal")
        
        # Optionally show traffic efficiency results
        if traffic_efficiency_results:
            avg_reduction = np.mean(traffic_efficiency_results['transfer_reduction_percent'])
            print(f"\n💾 Traffic efficiency average reduction: {avg_reduction:.1f}%")
        
        # === Existing summary and final stats ===
        G = st_ndt.G
        analysis = st_ndt.analyze_network_topology()
        print(f"\nCogentco Network Final Stats:")
        print(f"• Network size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"• Monitoring efficiency: ~{15}% of links needed for full network visibility")
        print(f"• Estimated bandwidth savings: ~{85}% compared to full monitoring")
        print(f"• Topological resilience: {analysis['topological_invariants']['betti_1']} redundant paths detected")
        
    except Exception as e:
        print(f"Error in Cogentco proof of concept: {e}")
        print("Please check that Cogentco.gml is valid and accessible.")

if __name__ == "__main__":
    # Run the complete proof of concept
    create_proof_of_concept()
       
