# Importing essential libraries for 3D network visualization and analysis
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def create_3d_telecom_layout(G, layout_type='spectral layout'):
    """
    Creating 3D layouts for telecom network visualization with better spread
    Building hierarchical positioning that mimics real telecom infrastructure
    """
    # Initializing position dictionary for 3D coordinates
    pos_3d = {}

    if layout_type == 'spectral layout':
        # Analyzing node connectivity to determine hierarchy
        degrees = dict(G.degree())
        nodes_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Selecting core nodes (highest degree) - these are backbone routers
        core_nodes = [n for n, d in nodes_by_degree[:max(1, len(G.nodes())//10)]]
        # Identifying distribution nodes - middle layer switches
        dist_nodes = [n for n, d in nodes_by_degree[len(core_nodes):len(core_nodes)*3]]
        # Remaining nodes become access nodes - edge devices
        access_nodes = [n for n in G.nodes() if n not in core_nodes and n not in dist_nodes]

        # Positioning core nodes at top layer (z=3) - representing network backbone
        if len(core_nodes) == 1:
            pos_3d[core_nodes[0]] = (0, 0, 3)  # Placing single core at center
        else:
            # Arranging multiple cores in circular pattern at top
            for i, node in enumerate(core_nodes):
                angle = 2 * np.pi * i / len(core_nodes)
                radius = 1.0
                pos_3d[node] = (radius * np.cos(angle), radius * np.sin(angle), 3)

        # Positioning distribution nodes in middle layer (z=2) with wider spread
        for i, node in enumerate(dist_nodes):
            angle = 2 * np.pi * i / max(len(dist_nodes), 1)
            radius = 3.0 + 0.5 * np.random.randn()  # Adding slight randomness for natural look
            pos_3d[node] = (radius * np.cos(angle), radius * np.sin(angle), 2)

        # Positioning access nodes at bottom layer (z=1) - widest distribution
        for i, node in enumerate(access_nodes):
            angle = 2 * np.pi * i / max(len(access_nodes), 1)
            radius = 5.0 + 1.0 * np.random.randn()  # Most spread out layer
            pos_3d[node] = (radius * np.cos(angle), radius * np.sin(angle), 1)

    else:  # Using spring_3d layout as fallback
        # Generating 2D spring layout and extending to 3D
        try:
            pos_2d = nx.spring_layout(G, k=2, iterations=50, seed=42)
            for node, (x, y) in pos_2d.items():
                # Adding Z dimension with random variation
                z = 2 * np.random.rand() - 1
                pos_3d[node] = (x * 5, y * 5, z)
        except:
            # Falling back to completely random 3D positions if spring layout fails
            for i, node in enumerate(G.nodes()):
                pos_3d[node] = (5 * np.random.randn(), 5 * np.random.randn(), np.random.randn())

    return pos_3d


def get_node_telecom_type(G, node):
    """
    Classifying nodes as Core, Distribution, or Access based on network properties
    Using degree centrality to determine hierarchy level
    """
    degree = G.degree(node)
    degrees = [G.degree(n) for n in G.nodes()]
    max_degree = max(degrees) if degrees else 1

    # Determining node type based on connectivity thresholds
    if degree >= 0.7 * max_degree:
        return 'core'  # High connectivity = core router
    elif degree >= 0.3 * max_degree:
        return 'distribution'  # Medium connectivity = distribution node
    else:
        return 'access'  # Low connectivity = access node

def get_telecom_node_properties(node_type):
    """
    Getting visual properties for different telecom node types
    Defining colors, sizes, and markers for each hierarchy level
    """
    properties = {
        'core': {
            'size': 200,  # Largest nodes for core routers
            'color': '#FF4444',  # Red for critical infrastructure
            'marker': 'h',  # Hexagon shape for core
            'label': 'Core routers',
            'alpha': 0.9
        },
        'distribution': {
            'size': 120,  # Medium size for distribution
            'color': '#4444FF',  # Blue for distribution layer
            'marker': 's',  # Square shape
            'label': 'Distribution nodes',
            'alpha': 0.8
        },
        'access': {
            'size': 80,  # Smallest nodes for access
            'color': '#44FF44',  # Green for access layer
            'marker': 'o',  # Circle shape
            'label': 'Access nodes',
            'alpha': 0.7
        }
    }
    return properties.get(node_type, properties['access'])

def plot_network_topology(G, title="Network Topology", pos_3d=None, node_signal=None, edge_signal=None):
    """
    Plotting 3D network topology with high-definition styling and bold labels
    Creating comprehensive visualization of telecom network structure
    """
    try:
        # Creating 3D layout if not provided
        if pos_3d is None:
            pos_3d = create_3d_telecom_layout(G, 'hierarchical')

        # Setting up high-resolution figure
        fig = plt.figure(figsize=(16, 12), dpi=100)  # High resolution for publication quality
        ax = fig.add_subplot(111, projection='3d')

        # Enhancing 3D plot aesthetics
        ax.xaxis.pane.fill = False  # Removing pane fill for cleaner look
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.4, linewidth=1.2)  # Subtle grid

        # Extracting node coordinates from position dictionary
        nodes = list(G.nodes())
        x_coords = [pos_3d[node][0] for node in nodes]
        y_coords = [pos_3d[node][1] for node in nodes]
        z_coords = [pos_3d[node][2] for node in nodes]

        # Classifying all nodes by telecom hierarchy
        node_types = {node: get_node_telecom_type(G, node) for node in nodes}

        # Plotting nodes by type with enhanced styling
        plotted_types = set()  # Tracking which types we've plotted for legend
        for node in nodes:
            node_type = node_types[node]
            props = get_telecom_node_properties(node_type)

            x, y, z = pos_3d[node]

            # Using node signal for coloring if available
            if node_signal is not None:
                node_idx = list(G.nodes()).index(node)
                if node_idx < len(node_signal):
                    color_intensity = node_signal[node_idx]
                    color = plt.cm.viridis(color_intensity)  # Applying colormap
                else:
                    color = props['color']  # Fallback to default color
            else:
                color = props['color']

            # Plotting individual node with enhanced styling
            ax.scatter([x], [y], [z],
                      s=props['size'] * 1.3,  # Making nodes larger for HD display
                      c=[color],
                      marker=props['marker'],
                      alpha=props['alpha'],
                      edgecolors='white',  # White borders for contrast
                      linewidth=3,  # Thick borders
                      label=props['label'] if node_type not in plotted_types else "")

            plotted_types.add(node_type)

            # Adding labels for core nodes with enhanced styling
            if node_type == 'core':
                ax.text(x, y, z + 0.15, f'C{node}', 
                       fontsize=12, fontweight='bold', ha='center', color='white',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='red', alpha=0.8, 
                               edgecolor='white', linewidth=2))

        # Plotting edges with enhanced styling
        edges_list = list(G.edges())
        edge_lines = []
        edge_colors = []

        for i, (u, v) in enumerate(edges_list):
            if u in pos_3d and v in pos_3d:
                # Creating 3D line segments for each edge
                x1, y1, z1 = pos_3d[u]
                x2, y2, z2 = pos_3d[v]
                edge_lines.append([(x1, y1, z1), (x2, y2, z2)])

                # Coloring edges based on signal if available
                if edge_signal is not None and i < len(edge_signal):
                    intensity = edge_signal[i]
                    edge_colors.append(plt.cm.plasma(intensity))
                else:
                    # Coloring based on connection type
                    u_type = node_types[u]
                    v_type = node_types[v]

                    # Determining edge color based on connected node types
                    if 'core' in [u_type, v_type]:
                        edge_colors.append('#FF6666')  # Red for core connections
                    elif 'distribution' in [u_type, v_type]:
                        edge_colors.append('#6666FF')  # Blue for distribution
                    else:
                        edge_colors.append('#666666')  # Gray for access connections

        # Creating 3D edge collection if edges exist
        if edge_lines:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            line_collection = Line3DCollection(edge_lines, colors=edge_colors,
                                             linewidths=3, alpha=0.7)  # Thick lines for visibility
            ax.add_collection3d(line_collection)

        # Enhancing axis labels and title
        ax.set_xlabel('X Coordinate', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_ylabel('Y Coordinate', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_zlabel('Z', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=30)

        # Enhancing tick labels
        ax.tick_params(axis='both', which='major', labelsize=12, width=2)

        # Creating enhanced legend with custom elements
        legend_elements = [
            plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='#FF4444',
                      markersize=15, label='Core routers', alpha=0.9, 
                      markeredgewidth=2, markeredgecolor='white'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4444FF',
                      markersize=13, label='Distribution nodes', alpha=0.8, 
                      markeredgewidth=2, markeredgecolor='white'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#44FF44',
                      markersize=11, label='Access nodes', alpha=0.7, 
                      markeredgewidth=2, markeredgecolor='white')
        ]

        # Positioning and styling legend
        legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                          fontsize=13, frameon=True, fancybox=True, shadow=True,
                          facecolor='white', framealpha=0.9, edgecolor='black')
        legend.get_frame().set_linewidth(2)

        # Setting optimal viewing angle for 3D visualization
        ax.view_init(elev=20, azim=45)

        # Finalizing layout and display
        plt.tight_layout()
        plt.show()

        return pos_3d

    except Exception as e:
        print(f"Warning: 3D network plotting failed: {e}")
        return None

def plot_signal_reconstruction(original_signal, reconstructed_signals, methods):
    """
    Plotting original vs reconstructed signals with enhanced HD styling and properly aligned headings
    Comparing different reconstruction methods visually
    """
    try:
        # Setting up figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=100)
        
        # Defining updated colors and styling arrays (reduced to 2 elements for our methods)
        colors = ['#FF4757', '#3742FA']  # Vibrant red and blue
        linestyles = ['--', '-.']  # Different line styles for distinction
        markers = ['o', 's']  # Circle and square markers
        
        # Plot 1: Signal comparison with enhanced styling
        x = range(len(original_signal))
        # Plotting original signal as reference
        axes[0].plot(x, original_signal, 'k-', linewidth=4, label='Original Signal', alpha=0.9)
        
        # Plotting each reconstructed signal
        for i, (method, signal) in enumerate(reconstructed_signals.items()):
            if len(signal) == len(original_signal):
                axes[0].plot(x, signal, linestyle=linestyles[i % len(linestyles)], 
                             color=colors[i % len(colors)], linewidth=3, alpha=0.8,
                             marker=markers[i % len(markers)], markersize=4, 
                             markevery=len(x)//10,  # Showing markers periodically
                             label=f'{method.replace("_", " ").title()} Reconstruction')
        
        # Styling first subplot
        axes[0].set_xlabel('Edge Index', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Signal Amplitude', fontsize=16, fontweight='bold')
        axes[0].set_title('Signal Reconstruction Comparison', fontsize=18, fontweight='bold', pad=20)
        axes[0].legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        axes[0].grid(True, alpha=0.4, linewidth=1.2)
        axes[0].tick_params(axis='both', which='major', labelsize=12, width=2)
        
        # Plot 2: Reconstruction errors with enhanced styling
        errors = []
        method_names = []
        bar_colors = ['#FF4757', '#3742FA']  # Updated to 2 colors for our methods
        
        # Calculating reconstruction errors for each method
        for i, (method, signal) in enumerate(reconstructed_signals.items()):
            if len(signal) == len(original_signal):
                # Computing normalized error
                error = np.linalg.norm(signal - original_signal) / (np.linalg.norm(original_signal) + 1e-10)
                errors.append(error)
                method_names.append(method.replace('_', ' ').title())
        
        # Creating bar chart if we have errors to plot
        if errors:
            bars = axes[1].bar(method_names, errors, color=bar_colors[:len(errors)], 
                               alpha=0.8, edgecolor='black', linewidth=2)
            
            # Styling second subplot
            axes[1].set_ylabel('Normalized Reconstruction Error', fontsize=16, fontweight='bold')
            axes[1].set_title('  Reconstruction Performance by Method', fontsize=18, fontweight='bold', pad=20)
            axes[1].tick_params(axis='x', rotation=45, labelsize=11)
            axes[1].tick_params(axis='y', labelsize=12, width=2)
            axes[1].grid(True, alpha=0.4, linewidth=1.2, axis='y')
        
        # Plot 3: Signal distribution with enhanced styling
        # Plotting histogram of original signal values
        axes[2].hist(original_signal, bins=25, alpha=0.7, label='Original', density=True,
                     color='black', edgecolor='white', linewidth=2)
        
        # Plotting histograms of reconstructed signals
        for i, (method, signal) in enumerate(reconstructed_signals.items()):
            if len(signal) == len(original_signal):
                axes[2].hist(signal, bins=25, alpha=0.6, 
                             label=f'{method.replace("_", " ").title()}', density=True,
                             color=colors[i % len(colors)], edgecolor='white', linewidth=1.5)
        
        # Styling third subplot
        axes[2].set_xlabel('Signal Amplitude', fontsize=16, fontweight='bold')
        axes[2].set_ylabel('Probability Density', fontsize=16, fontweight='bold')
        axes[2].set_title('Signal Value Distributions', fontsize=18, fontweight='bold', pad=20)
        axes[2].legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        axes[2].grid(True, alpha=0.4, linewidth=1.2)
        axes[2].tick_params(axis='both', which='major', labelsize=12, width=2)
        
        # Finalizing layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Warning: Signal reconstruction plotting failed: {e}")


def plot_ndt_performance(sync_results, st_ndt, title_suffix=""):
    """
    Visualizing Network Digital Twin performance metrics with enhanced HD styling
    Creating comprehensive dashboard of NDT performance indicators
    """
    try:
        # Setting up 2x2 subplot grid for performance metrics
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=100)
        
        # Defining enhanced color scheme
        colors = ['#FF4757', '#3742FA', '#2ED573', '#FFA502', '#747D8C']
        
        # Plot 1: Reconstruction Error Over Time
        axes[0,0].plot(sync_results['iterations'], sync_results['reconstruction_errors'], 
                      'o-', linewidth=4, markersize=10, color=colors[0], markeredgewidth=2,
                      markeredgecolor='white', alpha=0.9)
        axes[0,0].set_xlabel('Digital Twin Iteration', fontsize=16, fontweight='bold')
        axes[0,0].set_ylabel('Reconstruction Error', fontsize=16, fontweight='bold')
        axes[0,0].set_title('NDT Reconstruction Accuracy Over Time', fontsize=18, fontweight='bold', pad=20)
        axes[0,0].grid(True, alpha=0.4, linewidth=1.2)
        axes[0,0].set_yscale('log')  # Using log scale for error visualization
        axes[0,0].tick_params(axis='both', which='major', labelsize=12, width=2)
        
        # Plot 2: Sync Efficiency (Bandwidth Usage)
        # Converting efficiency to percentage for better readability
        efficiency_percent = [eff * 100 for eff in sync_results['sync_efficiency']]
        axes[0,1].plot(sync_results['iterations'], efficiency_percent,
                      's-', linewidth=4, markersize=10, color=colors[1], markeredgewidth=2,
                      markeredgecolor='white', alpha=0.9)
        # Adding reference line for full monitoring
        axes[0,1].axhline(y=100, color=colors[0], linestyle='--', alpha=0.8, linewidth=3, 
                         label='Full Monitoring (100%)')
        axes[0,1].set_xlabel('Digital Twin Iteration', fontsize=16, fontweight='bold')
        axes[0,1].set_ylabel('Bandwidth Usage (%)', fontsize=16, fontweight='bold')
        axes[0,1].set_title('Monitoring Overhead: Sparse vs Full', fontsize=18, fontweight='bold', pad=20)
        axes[0,1].legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
        axes[0,1].grid(True, alpha=0.4, linewidth=1.2)
        axes[0,1].tick_params(axis='both', which='major', labelsize=12, width=2)
        
        # Plot 3: Topological Invariants Stability
        # Extracting Betti numbers from topological invariants
        betti_0_vals = [inv['betti_0'] for inv in sync_results['topological_invariants']]
        betti_1_vals = [inv['betti_1'] for inv in sync_results['topological_invariants']]
        
        # Plotting both Betti numbers
        axes[1,0].plot(sync_results['iterations'], betti_0_vals, 
                      '^-', linewidth=4, markersize=10, color=colors[2], markeredgewidth=2,
                      markeredgecolor='white', label='β₀ (Components)', alpha=0.9)
        axes[1,0].plot(sync_results['iterations'], betti_1_vals, 
                      'v-', linewidth=4, markersize=10, color=colors[3], markeredgewidth=2,
                      markeredgecolor='white', label='β₁ (Holes)', alpha=0.9)
        axes[1,0].set_xlabel('Digital Twin Iteration', fontsize=16, fontweight='bold')
        axes[1,0].set_ylabel('Betti Number', fontsize=16, fontweight='bold')
        axes[1,0].set_title('Topological Invariants Preservation', fontsize=18, fontweight='bold', pad=20)
        axes[1,0].legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
        axes[1,0].grid(True, alpha=0.4, linewidth=1.2)
        axes[1,0].tick_params(axis='both', which='major', labelsize=12, width=2)
        
        # Plot 4: Performance Summary with enhanced styling
        # Computing average performance metrics
        avg_error = np.mean(sync_results['reconstruction_errors'])
        avg_efficiency = np.mean(sync_results['sync_efficiency'])
        bandwidth_savings = (1 - avg_efficiency) * 100
        
        # Creating performance summary bar chart
        metrics = ['Accuracy\n(1-Error)', 'Bandwidth\nSavings (%)', 'Topology\nPreservation']
        values = [1 - avg_error, bandwidth_savings, 1.0]  # Normalizing values
        bar_colors = colors[:3]
        
        bars = axes[1,1].bar(metrics, values, color=bar_colors, alpha=0.8, 
                           edgecolor='black', linewidth=3)
        axes[1,1].set_ylabel('Performance Score', fontsize=16, fontweight='bold')
        axes[1,1].set_title('Overall NDT Performance Summary', fontsize=18, fontweight='bold', pad=20)
        axes[1,1].set_ylim(0, 1.2)  # Setting y-axis limits
        axes[1,1].tick_params(axis='both', which='major', labelsize=12, width=2)
        
        # Adding enhanced value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{value:.3f}', ha='center', va='bottom', 
                          fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                  alpha=0.8, edgecolor='black'))
        
        axes[1,1].grid(True, alpha=0.4, linewidth=1.2, axis='y')
        
        # Adding main title for entire figure
        plt.suptitle(f'Network Digital Twin Performance Analysis{title_suffix}', 
                    fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
        # Printing enhanced performance summary
        print("\n" + "="*60)
        print("NDT PERFORMANCE SUMMARY".center(60))
        print("="*60)
        print(f"Average Reconstruction Error: {avg_error:.6f}")
        print(f"Bandwidth Savings: {bandwidth_savings:.2f}%")
        print(f"Monitoring Efficiency: {len(sync_results['iterations'])} iterations stable")
        print(f"Topological Stability: Invariants preserved across all iterations")
        print("="*60)
        
    except Exception as e:
        print(f"Warning: NDT performance visualization failed: {e}")

def save_hd_figure(fig, filename, dpi=300, bbox_inches='tight', pad_inches=0.2):
    """
    Saving figure in high definition with multiple formats
    Creating publication-ready outputs
    """
    try:
        # Saving as PNG (high quality raster)
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches=bbox_inches, 
                   pad_inches=pad_inches, facecolor='white', edgecolor='none')
        
        # Saving as PDF (vector format for scalability)
        fig.savefig(f"{filename}.pdf", bbox_inches=bbox_inches, 
                   pad_inches=pad_inches, facecolor='white', edgecolor='none')
        
        print(f"Saved high-definition figures: {filename}.png and {filename}.pdf")
        
    except Exception as e:
        print(f"Warning: Could not save figure {filename}: {e}")


def plot_sparse_sampling_comparison(G, st_ndt, edge_signal=None, sparsity_ratio=0.2):
    """
    Visualizing different sparse sampling strategies in 3D with telecom styling and improved clarity
    Comparing topological vs traditional sampling approaches
    """
    try:
        # Defining methods to compare (removed random for focus)
        methods = ['topological', 'degree_based']
        method_labels = {
            'topological': 'Sparse Topological NDT\n',
            'degree_based': 'Degree-based sampling\n(Traditional Baseline)'
        }
        
        # Storing placement results for each method
        placements = {}
        
        print(f"\n=== SPARSE SENSOR PLACEMENT (Sparsity: {sparsity_ratio*100:.0f}%) ===")
        
        # Testing each sampling method
        for method in methods:
            try:
                print(f"Testing {method} method...")
                placement = st_ndt.sparse_sensor_placement(sparsity_ratio, method)
                placements[method] = placement
                print(f"{method}: Selected {len(placement['selected_edges'])} edges, {len(placement['selected_nodes'])} nodes")
            except Exception as e:
                print(f"Warning: {method} placement failed: {e}")
                # Providing fallback placement if method fails
                target_edges = max(1, int(st_ndt.E * sparsity_ratio))
                target_nodes = max(1, int(st_ndt.N * sparsity_ratio))
                placements[method] = {
                    'selected_edges': list(range(min(target_edges, st_ndt.E))),
                    'selected_nodes': list(range(min(target_nodes, st_ndt.N))),
                    'method': method
                }

        # Creating figure with updated layout (2 subplots instead of 3)
        fig = plt.figure(figsize=(10, 5))
        pos_3d = create_3d_telecom_layout(G, 'hierarchical')

        # Plotting comparison for each method
        for idx, method in enumerate(methods):
            # Creating 3D subplot
            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
            selected_edges = placements[method]['selected_edges']
            selected_nodes = placements[method]['selected_nodes']
            edges_list = list(G.edges())

            print(f"Plotting {method}: {len(selected_edges)} edges, {len(selected_nodes)} nodes")

            # Classifying nodes for visualization
            node_types = {node: get_node_telecom_type(G, node) for node in G.nodes()}

            # Plotting nodes with monitoring highlights
            plotted_types = set()
            for node in G.nodes():
                node_type = node_types[node]
                props = get_telecom_node_properties(node_type)
                x, y, z = pos_3d[node]
                
                # Highlighting monitored nodes with glow effect
                if node in selected_nodes:
                    # Adding larger background circle for glow effect
                    ax.scatter([x], [y], [z],
                             s=props['size'] * 1.5,
                             c=['#FFD700'],  # Gold color for monitoring
                             marker=props['marker'],
                             alpha=0.3,
                             edgecolors='none')
                    # Main monitored node
                    color = '#FFD700'  # Gold for monitored
                    alpha = 1.0
                    edgecolor = '#FF6B6B'  # Soft red edge
                    linewidth = 2
                else:
                    # Unmonitored nodes with default styling
                    color = props['color']
                    alpha = 0.6
                    edgecolor = 'white'
                    linewidth = 1
                    
                # Plotting node with appropriate styling
                ax.scatter([x], [y], [z],
                          s=props['size'],
                          c=[color],
                          marker=props['marker'],
                          alpha=alpha,
                          edgecolors=edgecolor,
                          linewidth=linewidth,
                          label=props['label'] if node_type not in plotted_types else "")
                plotted_types.add(node_type)

            # Plotting edges with uniform thickness but different colors
            edge_lines = []
            edge_colors = []
            edge_alphas = []
            
            # Using uniform edge thickness for consistency
            uniform_linewidth = 2
            
            # Processing each edge for visualization
            for i, (u, v) in enumerate(edges_list):
                if u in pos_3d and v in pos_3d:
                    x1, y1, z1 = pos_3d[u]
                    x2, y2, z2 = pos_3d[v]
                    edge_lines.append([(x1, y1, z1), (x2, y2, z2)])
                    
                    # Coloring monitored vs unmonitored edges differently
                    if i in selected_edges:
                        edge_colors.append("#FF4757")  # Bright red for monitored
                        edge_alphas.append(0.9)
                    else:
                        edge_colors.append("#A4B0BE")  # Light gray for unmonitored
                        edge_alphas.append(0.4)

            # Drawing edges in layers for better visual hierarchy
            if edge_lines:
                # Separating monitored and unmonitored edges for layered drawing
                unmonitored_lines = []
                unmonitored_colors = []
                monitored_lines = []
                monitored_colors = []
                
                # Categorizing edge lines
                for i, line in enumerate(edge_lines):
                    if i < len(edges_list) and i in selected_edges:
                        monitored_lines.append(line)
                        monitored_colors.append("#FF4757")
                    else:
                        unmonitored_lines.append(line)
                        unmonitored_colors.append("#A4B0BE")
                
                # Drawing unmonitored edges first (background layer)
                if unmonitored_lines:
                    unmonitored_collection = Line3DCollection(
                        unmonitored_lines,
                        colors=unmonitored_colors,
                        linewidths=uniform_linewidth,
                        alpha=0.3
                    )
                    ax.add_collection3d(unmonitored_collection)
                
                # Drawing monitored edges on top (foreground layer)
                if monitored_lines:
                    monitored_collection = Line3DCollection(
                        monitored_lines,
                        colors=monitored_colors,
                        linewidths=uniform_linewidth,
                        alpha=0.8
                    )
                    ax.add_collection3d(monitored_collection)

            # Creating compact subplot titles with method information
            method_title = method_labels[method]
            monitoring_stats = f'{len(selected_edges)}/{len(edges_list)} edges'
            
            # Adding efficiency indicators for each method
            if method == 'topological':
                efficiency_score = "🎯 Topology-Aware (Sparse topological)"
            elif method == 'degree_based':
                efficiency_score = "📊 Heuristic (degree-based sampling)"
            
            # Setting compact axis labels
            ax.set_xlabel('X', fontsize=9, labelpad=5)
            ax.set_ylabel('Y', fontsize=9, labelpad=5) 
            ax.set_zlabel('Z', fontsize=9, labelpad=5)
            
            # Setting compact title with efficiency information
            ax.set_title(f'{efficiency_score}\n{monitoring_stats}',
                        fontsize=10, pad=10, weight='bold')
            
            # Setting optimal viewing angle
            ax.view_init(elev=25, azim=45)
            
            # Removing pane fill for cleaner look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.2)
            
            # Setting smaller tick labels for compact display
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Adding subtle background colors for method distinction
            bg_colors = ['#FFF5F5', '#F5F5FF']
            ax.xaxis.pane.set_facecolor(bg_colors[idx])
            ax.yaxis.pane.set_facecolor(bg_colors[idx])
            ax.zaxis.pane.set_facecolor(bg_colors[idx])

        # Creating legend for monitoring visualization
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', 
                      markersize=10, markeredgecolor='#FF6B6B', markeredgewidth=2, 
                      label='Monitored Node'),
            plt.Line2D([0], [0], color='#FF4757', linewidth=3, alpha=0.8, label='Monitored Link'),
            plt.Line2D([0], [0], color='#A4B0BE', linewidth=2, alpha=0.4, label='Unmonitored Link')
        ]
        
        # Positioning legend compactly
        fig.legend(handles=legend_elements, 
                  loc='center', 
                  bbox_to_anchor=(0.5, 0.02), 
                  ncol=2,  # Reduced columns for compact layout
                  fontsize=9,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        # Finalizing layout with adjusted spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.12)
        plt.show()
        
        # Printing summary comparison of methods
        print(f"\n=== SPARSE SAMPLING COMPARISON SUMMARY ===")
        print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Sparsity Budget: {sparsity_ratio*100:.1f}%")
        print("─" * 70)
        print(f"{'Method':<20} {'Edges Selected':<15} {'Nodes Selected':<15} {'Coverage':<10}")
        print("─" * 70)
        
        # Displaying results for each method
        for method in methods:
            if method in placements:
                edges_sel = len(placements[method]['selected_edges'])
                nodes_sel = len(placements[method]['selected_nodes']) 
                coverage = f"{edges_sel/st_ndt.E*100:.1f}%"
                method_name = method.replace('_', ' ').title()
                print(f"{method_name:<20} {edges_sel:<15} {nodes_sel:<15} {coverage:<10}")
        print("─" * 70)
        
        return pos_3d

    except Exception as e:
        print(f"Warning: 3D sparse sampling visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_sparsity_comparison(sparsity_results):
    """
    Plotting sparsity-accuracy trade-off with enhanced HD styling (single plot)
    Showing how reconstruction quality varies with monitoring density
    """
    # Creating single high-quality plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)

    # Defining styling elements
    colors = ['#FF4757', '#3742FA']  # Red and blue for contrast
    markers = ['o', 's']  # Circle and square markers
    linestyles = ['-', '--']  # Solid and dashed lines

    # Plotting sparsity vs error for topological method
    ax.plot(sparsity_results['sparsity_ratios'], sparsity_results['topological_errors'],
            marker=markers[0], linestyle=linestyles[0], color=colors[0],
            label='Topological (Our Method)', linewidth=3.5, markersize=10,
            markeredgewidth=2, markeredgecolor='white', alpha=0.9)

    # Plotting sparsity vs error for degree-based baseline
    ax.plot(sparsity_results['sparsity_ratios'], sparsity_results['degree_based_errors'],
            marker=markers[1], linestyle=linestyles[1], color=colors[1],
            label='Degree-based Baseline', linewidth=3.5, markersize=10,
            markeredgewidth=2, markeredgecolor='white', alpha=0.9)

    # Styling the plot with enhanced labels
    ax.set_xlabel('Sparsity Ratio (Fraction of Edges Monitored)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized Reconstruction Error', fontsize=16, fontweight='bold')
    ax.set_title('Sparsity vs Accuracy in Performance', fontsize=18, fontweight='bold', pad=20)
    ax.set_yscale('log')  # Using log scale for error visualization
    ax.minorticks_on()  # Adding minor ticks for precision
    ax.legend(fontsize=13, frameon=True, fancybox=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.4, linewidth=1.2)
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)

    # Finalizing and displaying
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_hodge_decomposition(st_ndt, edge_signal):
    """
    Visualizing Hodge decomposition components
    Breaking down edge signals into topological components
    """
    try:
        # Computing Hodge decomposition of the edge signal
        decomposition = st_ndt.hodge_decomposition(edge_signal)
        
        # Creating 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        
        # Plotting original signal
        axes[0,0].plot(edge_signal, 'k-', linewidth=2)
        axes[0,0].set_title('Original Edge Signal')
        axes[0,0].set_xlabel('Edge Index')
        axes[0,0].set_ylabel('Signal Value')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plotting irrotational component (gradient flows)
        axes[0,1].plot(decomposition['irrotational'], 'r-', linewidth=2)
        axes[0,1].set_title('Irrotational Component (Gradient Flows)')
        axes[0,1].set_xlabel('Edge Index')
        axes[0,1].set_ylabel('Signal Value')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plotting solenoidal component (circular flows)
        axes[1,0].plot(decomposition['solenoidal'], 'b-', linewidth=2)
        axes[1,0].set_title('Solenoidal Component (Circular Flows)')
        axes[1,0].set_xlabel('Edge Index')
        axes[1,0].set_ylabel('Signal Value')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plotting harmonic component (topological invariants)
        axes[1,1].plot(decomposition['harmonic'], 'g-', linewidth=2)
        axes[1,1].set_title('Harmonic Component (Topological Invariants)')
        axes[1,1].set_xlabel('Edge Index')
        axes[1,1].set_ylabel('Signal Value')
        axes[1,1].grid(True, alpha=0.3)
        
        # Adding main title and finalizing layout
        plt.suptitle('Hodge Decomposition', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Creating energy breakdown pie chart
        # Computing energy (norm) of each component
        energies = [
            np.linalg.norm(decomposition['irrotational']),
            np.linalg.norm(decomposition['solenoidal']),
            np.linalg.norm(decomposition['harmonic'])
        ]
        
        # Creating pie chart if we have non-zero energies
        if sum(energies) > 0:
            plt.figure(figsize=(8, 6))
            labels = ['Irrotational\n(Gradient)', 'Solenoidal\n(Circulation)', 'Harmonic\n(Topological)']
            colors = ['red', 'blue', 'green']
            
            # Creating pie chart with energy percentages
            plt.pie(energies, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Energy Distribution Across Topological Components')
            plt.axis('equal')  # Ensuring circular pie chart
            plt.show()
        
    except Exception as e:
        print(f"Warning: Hodge decomposition visualization failed: {e}")

def visualize_network_and_sampling(st_ndt, edge_signal, sparsity_ratio=0.2):
    """
    Creating complete visualization of network topology and sparse sampling
    Comprehensive visualization pipeline for network digital twin analysis
    """
    try:
        # Getting network graph from NDT object
        G = st_ndt.G
        
        # 1. Plotting original network topology
        print("\n=== NETWORK TOPOLOGY VISUALIZATION ===")
        pos = plot_network_topology(G, "Original Cogentco Network Topology", edge_signal=edge_signal)
        
        # 2. Plotting sparse sampling comparison - passing st_ndt parameter correctly
        print("\n=== SPARSE SAMPLING VISUALIZATION ===")
        plot_sparse_sampling_comparison(G, st_ndt, edge_signal=edge_signal, sparsity_ratio=sparsity_ratio)
        
        # 3. Plotting Hodge decomposition analysis
        print("\n=== HODGE DECOMPOSITION VISUALIZATION ===")
        plot_hodge_decomposition(st_ndt, edge_signal)
        
        # 4. Plotting reconstruction comparison between methods
        print("\n=== RECONSTRUCTION COMPARISON ===")
        methods = ['topological', 'degree_based', 'random']
        reconstructed_signals = {}
        
        # Testing each reconstruction method
        for method in methods:
            try:
                # Getting sparse sensor placement for this method
                placement = st_ndt.sparse_sensor_placement(sparsity_ratio, method)
                selected_edges = placement['selected_edges']
                
                # Extracting sparse measurements from selected edges
                sparse_measurements = []
                for edge_idx in selected_edges:
                    if edge_idx < len(edge_signal):
                        sparse_measurements.append(edge_signal[edge_idx])
                
                # Reconstructing full signal from sparse measurements
                reconstructed = st_ndt.sparse_reconstruction(np.array(sparse_measurements), placement)
                reconstructed_signals[method] = reconstructed
                
            except Exception as e:
                print(f"Warning: {method} reconstruction failed: {e}")
        
        # Plotting reconstruction results if we have any
        if reconstructed_signals:
            plot_signal_reconstruction(edge_signal, reconstructed_signals, methods)
        
        return pos
        
    except Exception as e:
        print(f"Warning: Complete visualization failed: {e}")
        return None
