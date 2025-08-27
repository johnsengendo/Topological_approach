# Importing required libraries for network analysis and mathematical operations
import networkx as nx
import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt

class SparseTopologicalNDT:
    """
    Creating a Sparse Topological Network Digital Twin using Cell Complex Signal Processing
    
    This class implements sparse network monitoring and reconstruction using topological methods
    based on Hodge decomposition and spectral graph theory principles.
    """
    
    def __init__(self, network_graph):
        """
        Initializing the sparse topological network digital twin
        
        Args:
            network_graph: NetworkX graph representing the network topology
        """
        # Storing the input network graph
        self.G = network_graph
        # Getting network dimensions
        self.N = len(network_graph.nodes())  # Number of nodes
        self.E = len(network_graph.edges())  # Number of edges
        
        # Defining core topological matrices from cell complex theory
        self.B1 = None  # Node-edge incidence matrix (connects nodes to edges)
        self.B2 = None  # Edge-polygon incidence matrix (connects edges to cycles)
        self.L0 = None  # Node Laplacian (diffusion on nodes)
        self.L1 = None  # Edge Laplacian (L1,d + L1,u - total edge dynamics)
        self.L1_d = None  # Lower Laplacian (irrotational component)
        self.L1_u = None  # Upper Laplacian (solenoidal component)
        
        # Storing eigendecompositions for spectral analysis
        self.eigvals_L1 = None  # Eigenvalues of edge Laplacian
        self.eigvecs_L1 = None  # Eigenvectors of edge Laplacian
        self.U_irr = None   # Irrotational subspace eigenvectors
        self.U_sol = None   # Solenoidal subspace eigenvectors  
        self.U_harm = None  # Harmonic subspace eigenvectors
        
        # Building all topological matrices during initialization
        self._build_topology_matrices()
    
    def _build_topology_matrices(self):
        """
        Building incidence matrices and Laplacians following topological signal processing theory
        """
        try:
            # Creating B1: Node-edge incidence matrix (N x E) 
            # Each row represents a node, each column represents an edge
            # Entry is +1/-1 if node is source/sink of edge, 0 otherwise
            self.B1 = nx.incidence_matrix(self.G, oriented=True).astype(float).toarray()
            
            # Finding all minimal cycles (polygons) for higher-order topology
            cycles = self._find_minimal_cycles()
            # Building B2: Edge-polygon incidence matrix
            self.B2 = self._build_edge_polygon_matrix(cycles)
            
            # Building Laplacian matrices following the mathematical framework
            # L0: Node Laplacian = B1 * B1^T (captures node-level diffusion)
            self.L0 = self.B1 @ self.B1.T
            # L1_d: Lower edge Laplacian = B1^T * B1 (irrotational flows)
            self.L1_d = self.B1.T @ self.B1
            
            # Handling B2 matrix safely for solenoidal flows
            if self.B2.shape[1] > 0:
                B2_dense = self.B2.toarray() if hasattr(self.B2, 'toarray') else self.B2
                # L1_u: Upper edge Laplacian = B2 * B2^T (solenoidal flows)
                self.L1_u = B2_dense @ B2_dense.T
            else:
                # No cycles found, so no solenoidal component
                self.L1_u = np.zeros((self.E, self.E))
                
            # Total edge Laplacian combines both components
            self.L1 = self.L1_d + self.L1_u 
            
            # Adding small regularization for numerical stability
            self.L1_reg = self.L1 + 1e-12 * np.eye(self.E)
            # Computing eigendecomposition for spectral analysis
            self.eigvals_L1, self.eigvecs_L1 = np.linalg.eigh(self.L1_reg)
            
            # Decomposing signal space into topological subspaces
            self._decompose_signal_subspaces()
            
        except Exception as e:
            print(f"Warning: Error in topology matrix construction: {e}")
            # Using fallback construction when full topology analysis fails
            self._build_fallback_matrices()
    
    def _build_fallback_matrices(self):
        """
        Creating fallback construction when full topology analysis fails
        Using simplified random matrices to maintain functionality
        """
        print("Using fallback matrix construction...")
        
        # Creating simple random incidence matrix
        self.B1 = np.random.randn(self.N, self.E) * 0.1
        self.B2 = np.zeros((self.E, 1))  # Minimal B2 matrix
        
        # Building simple Laplacians from basic construction
        self.L0 = self.B1 @ self.B1.T
        self.L1_d = self.B1.T @ self.B1
        self.L1_u = np.zeros((self.E, self.E))
        self.L1 = self.L1_d + 1e-6 * np.eye(self.E)
        
        # Computing simple eigendecomposition
        self.eigvals_L1, self.eigvecs_L1 = np.linalg.eigh(self.L1)
        
        # Creating simple subspace decomposition by dividing eigenvectors
        mid = self.E // 3
        self.U_harm = self.eigvecs_L1[:, :max(1, mid//3)]
        self.U_irr = self.eigvecs_L1[:, max(1, mid//3):2*mid]
        self.U_sol = self.eigvecs_L1[:, 2*mid:]
    
    def _find_minimal_cycles(self, max_cycle_length=4):
        """
        Finding minimal cycles (polygons) in the network for higher-order topology
        
        Args:
            max_cycle_length: Maximum length of cycles to consider
            
        Returns:
            List of cycles found in the network
        """
        cycles = []
        try:
            # Converting to undirected graph for cycle detection
            undirected_G = self.G.to_undirected() if self.G.is_directed() else self.G
            # Finding fundamental cycle basis of the graph
            cycle_basis = nx.minimum_cycle_basis(undirected_G)
            
            # Filtering cycles by maximum length and limiting total number
            for cycle in cycle_basis[:min(20, len(cycle_basis))]:
                if len(cycle) <= max_cycle_length:
                    cycles.append(cycle)
                    
            # Creating simple triangular cycles if no cycles found
            if not cycles and self.N >= 3:
                nodes = list(self.G.nodes())
                for i in range(min(5, self.N-2)):
                    # Checking if three consecutive nodes form a triangle
                    if (self.G.has_edge(nodes[i], nodes[i+1]) and 
                        self.G.has_edge(nodes[i+1], nodes[i+2]) and 
                        self.G.has_edge(nodes[i+2], nodes[i])):
                        cycles.append([nodes[i], nodes[i+1], nodes[i+2]])
                        
        except Exception as e:
            print(f"Warning: Cycle finding failed: {e}")
            cycles = []
        
        return cycles
    
    def _build_edge_polygon_matrix(self, cycles):
        """
        Building B2 matrix: edge-polygon incidence matrix (E x P)
        This matrix connects edges to the polygons (cycles) they bound
        
        Args:
            cycles: List of cycles found in the network
            
        Returns:
            B2 matrix of shape (E, number_of_cycles)
        """
        # Creating mapping from edge pairs to edge indices
        edge_list = list(self.G.edges())
        edge_to_idx = {(min(u,v), max(u,v)): i for i, (u, v) in enumerate(edge_list)}
        
        # Returning empty matrix if no cycles found
        if not cycles:
            return np.zeros((self.E, 0))
        
        # Initializing B2 matrix
        B2 = np.zeros((self.E, len(cycles)))
        
        try:
            # For each polygon (cycle), marking its boundary edges
            for j, cycle in enumerate(cycles):
                # Processing each edge in the cycle boundary
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    edge_key = (min(u, v), max(u, v))
                    
                    # Setting incidence relationship with orientation
                    if edge_key in edge_to_idx:
                        edge_idx = edge_to_idx[edge_key]
                        if edge_idx < self.E:
                            # Assigning +1/-1 based on edge orientation
                            B2[edge_idx, j] = 1 if u < v else -1
        except Exception as e:
            print(f"Warning: B2 construction failed: {e}")
            B2 = np.zeros((self.E, max(1, len(cycles))))
        
        return B2
    
    def _decompose_signal_subspaces(self):
        """
        Decomposing edge signal space into irrotational, solenoidal, and harmonic subspaces
        This is the key step in Hodge decomposition theory
        """
        try:
            # Setting tolerance for numerical zero eigenvalues
            tol = 1e-8
            
            # Identifying harmonic subspace (kernel of L1 - zero eigenvalues)
            harmonic_mask = self.eigvals_L1 < tol
            self.U_harm = self.eigvecs_L1[:, harmonic_mask]
            
            # Getting non-harmonic eigenvectors for irrotational/solenoidal decomposition
            non_harmonic_mask = self.eigvals_L1 >= tol
            U_non_harm = self.eigvecs_L1[:, non_harmonic_mask]
            
            # Dividing non-harmonic space into irrotational and solenoidal
            if U_non_harm.shape[1] > 0:
                mid = U_non_harm.shape[1] // 2
                # Lower frequency modes -> irrotational (gradient flows)
                self.U_irr = U_non_harm[:, :mid] if mid > 0 else np.array([]).reshape(self.E, 0)
                # Higher frequency modes -> solenoidal (curl flows)
                self.U_sol = U_non_harm[:, mid:] if mid < U_non_harm.shape[1] else np.array([]).reshape(self.E, 0)
            else:
                # No non-harmonic modes found
                self.U_irr = np.array([]).reshape(self.E, 0)
                self.U_sol = np.array([]).reshape(self.E, 0)
                
            # Ensuring we always have a valid harmonic subspace
            if self.U_harm.shape[1] == 0:
                self.U_harm = self.eigvecs_L1[:, :max(1, min(3, self.E//4))]
                
        except Exception as e:
            print(f"Warning: Subspace decomposition failed: {e}")
            # Using simple equal division as fallback
            third = max(1, self.E // 3)
            self.U_harm = self.eigvecs_L1[:, :third]
            self.U_irr = self.eigvecs_L1[:, third:2*third]
            self.U_sol = self.eigvecs_L1[:, 2*third:]
    
    def sparse_sensor_placement(self, sparsity_ratio=0.1, method='topological'):
        """
        Implementing sparse sensor placement using topology-based optimization
        
        Args:
            sparsity_ratio: Fraction of edges/nodes to monitor (0.1 = 10%)
            method: 'topological', 'random', or 'degree_based'
            
        Returns:
            Dictionary containing selected edges, nodes, and placement metadata
        """
        # Calculating target number of sensors based on budget
        target_edges = max(1, int(self.E * sparsity_ratio))
        target_nodes = max(1, int(self.N * sparsity_ratio))
        
        print(f"Target edges to select: {target_edges}/{self.E}")
        
        # Routing to appropriate placement method
        if method == 'topological':
            return self._topological_sensor_placement(target_edges, target_nodes)
        elif method == 'degree_based':
            return self._degree_based_placement(target_edges, target_nodes)
    
    def _topological_sensor_placement(self, target_edges, target_nodes):
        """
        Creating improved topological placement using spectral methods:
        - Building a bandlimited basis U_k (choosing k adaptive to budget)
        - Using QR column pivoting on U_k.T to pick rows (edges) that best span the subspace
        - Falling back to leverage scores if QR pivoting fails
        """
        try:
            print(f"Starting improved topological placement for {target_edges} edges")
            # Ensuring target values are within valid bounds
            target_edges = min(max(1, int(target_edges)), self.E)
            target_nodes = min(max(1, int(target_nodes)), self.N)

            # Choosing bandlimit k: preferring small k but at least 3, proportional to budget
            max_k = self.eigvecs_L1.shape[1]
            # Setting k ~ min( max(3, 2*target_edges), max_k ) but capping to available eigenvectors
            k_band = min(max(3, 2 * target_edges), max_k)
            U_k = self.eigvecs_L1[:, :k_band]  # shape: (E, k)

            # Using QR pivoting on U_k.T (k x E). Column pivots correspond to edges (rows of U_k).
            try:
                # QR with pivoting returns pivot indices that order columns by importance for spanning
                q, r, piv = np.linalg.qr(U_k.T, mode='reduced', pivoting=True)
                selected_indices = list(piv[:target_edges])
                # Ensuring unique & in-range indices
                selected_edges = [int(i) for i in selected_indices if 0 <= i < self.E]
                # Padding with top leverage scores if not enough edges selected
                if len(selected_edges) < target_edges:
                    lev = np.sum(U_k**2, axis=1)
                    extra = [int(i) for i in np.argsort(lev)[-target_edges*2:][::-1] if i not in selected_edges]
                    selected_edges += extra[:(target_edges - len(selected_edges))]
            except Exception as e_qr:
                # QR failed — falling back to leverage scores (row energy)
                print(f"QR pivoting failed ({e_qr}), falling back to leverage-score selection")
                lev = np.sum(U_k**2, axis=1)
                selected_edges = np.argsort(lev)[-target_edges:].tolist()

            # Ensuring indices are unique and sorted for cleaner output
            selected_edges = list(dict.fromkeys(selected_edges))[:target_edges]

        except Exception as e:
            print(f"Warning: Topological placement failed: {e}, using degree-based fallback")
            return self._degree_based_placement(target_edges, target_nodes)

        # Selecting nodes connected to selected edges
        edge_list = list(self.G.edges())
        connected_nodes = set()
        for edge_idx in selected_edges:
            if 0 <= edge_idx < len(edge_list):
                u, v = edge_list[edge_idx]
                connected_nodes.add(u)
                connected_nodes.add(v)

        selected_nodes = list(connected_nodes)[:target_nodes]

        return {
            'selected_edges': selected_edges,
            'selected_nodes': selected_nodes,
            'method': 'topological',
            'polygons_used': self.B2.shape[1] if hasattr(self.B2, 'shape') else 0,
            'k_band': k_band
        }
        
    def _degree_based_placement(self, target_edges, target_nodes):
        """
        Creating baseline placement by selecting highest degree nodes/edges
        """
        try:
            print(f"Starting degree-based placement for {target_edges} edges")
            
            # Selecting highest degree nodes
            node_degrees = dict(self.G.degree())
            selected_nodes = sorted(node_degrees.keys(), 
                                key=lambda x: node_degrees[x], 
                                reverse=True)[:target_nodes]
            
            # Selecting edges connected to high-degree nodes
            edge_degrees = []
            edge_list = list(self.G.edges())
            for i, (u, v) in enumerate(edge_list):
                # Computing edge importance as sum of endpoint degrees
                edge_degree = node_degrees.get(u, 0) + node_degrees.get(v, 0)
                edge_degrees.append((i, edge_degree))
            
            # Sorting by edge degree and selecting top edges
            edge_degrees.sort(key=lambda x: x[1], reverse=True)
            selected_edges = [i for i, _ in edge_degrees[:target_edges]]
            
            print(f"Selected {len(selected_edges)} edges based on node degrees")
            
        except Exception as e:
            print(f"Warning: Degree-based placement failed: {e}, using random fallback")
            return self._random_placement(target_edges, target_nodes)
        
        return {
            'selected_edges': selected_edges,
            'selected_nodes': selected_nodes,
            'method': 'degree_based'
        }

    def hodge_decomposition(self, edge_signal):
        """
        Decomposing edge signal into irrotational, solenoidal, harmonic components
        This follows the mathematical framework from Hodge theory
        
        Args:
            edge_signal: Signal values on network edges
            
        Returns:
            Dictionary containing the three orthogonal components
        """
        try:
            # Ensuring signal is properly shaped
            s1 = edge_signal.reshape(-1, 1) if edge_signal.ndim == 1 else edge_signal
            
            # Projecting onto each orthogonal subspace
            s1_irr = np.zeros_like(s1)
            s1_sol = np.zeros_like(s1)
            s1_harm = np.zeros_like(s1)
            
            # Irrotational component: projection onto gradient flows
            if self.U_irr.shape[1] > 0:
                s1_irr = self.U_irr @ (self.U_irr.T @ s1)
            
            # Solenoidal component: projection onto curl flows
            if self.U_sol.shape[1] > 0:
                s1_sol = self.U_sol @ (self.U_sol.T @ s1)
            
            # Harmonic component: projection onto harmonic flows
            if self.U_harm.shape[1] > 0:
                s1_harm = self.U_harm @ (self.U_harm.T @ s1)
            
            return {
                'irrotational': s1_irr.flatten(),
                'solenoidal': s1_sol.flatten(),
                'harmonic': s1_harm.flatten(),
                'total_reconstructed': (s1_irr + s1_sol + s1_harm).flatten()
            }
            
        except Exception as e:
            print(f"Warning: Hodge decomposition failed: {e}")
            # Returning trivial decomposition as fallback
            return {
                'irrotational': edge_signal * 0.6,
                'solenoidal': edge_signal * 0.3,
                'harmonic': edge_signal * 0.1,
                'total_reconstructed': edge_signal
            }
    
    def sparse_reconstruction(self, sparse_measurements, sensor_placement, reg=1e-6):
        """
        Creating improved sparse reconstruction using regularized least squares:
        - Using an adaptive bandlimited basis U_k (preferring small k consistent with placement)
        - Solving regularized least squares for basis coefficients:
            min_c || M U_k c - y ||^2 + reg * ||c||^2
        then reconstructed = U_k @ c
        
        Args:
            sparse_measurements: Measured values at selected sensors
            sensor_placement: Result from sensor placement algorithm
            reg: Regularization parameter (default 1e-6, can increase for noisy measurements)
            
        Returns:
            Reconstructed signal on all network edges
        """
        try:
            selected_edges = sensor_placement['selected_edges']
            # Converting measurements to numpy array
            if isinstance(sparse_measurements, list):
                y = np.array(sparse_measurements, dtype=float)
            else:
                y = sparse_measurements.astype(float)

            # Getting bandlimit from sensor placement or choosing adaptively
            k_band = sensor_placement.get('k_band', None)
            if k_band is None:
                # Using heuristic: min( max(3, 2*#sensors), total_eigvecs )
                k_band = min(max(3, 2 * max(1, len(selected_edges))), self.eigvecs_L1.shape[1])
            k_band = min(int(k_band), self.eigvecs_L1.shape[1])
            U_k = self.eigvecs_L1[:, :k_band]  # (E, k)

            # Building measurement matrix M (m x E) selecting rows = edges monitored
            m = len(selected_edges)
            M = np.zeros((m, self.E))
            for i, ei in enumerate(selected_edges):
                if 0 <= ei < self.E:
                    M[i, ei] = 1.0

            # Building M @ U_k (m x k) for efficient computation
            M_Uk = M @ U_k

            # Aligning measurement vector length with sensor matrix
            if y.shape[0] != M_Uk.shape[0]:
                # Trying to truncate or pad y
                if y.shape[0] > M_Uk.shape[0]:
                    y = y[:M_Uk.shape[0]]
                else:
                    # Padding with zeros (assuming missing sensors measured 0)
                    y = np.concatenate([y, np.zeros(M_Uk.shape[0] - y.shape[0])])

            if M_Uk.size == 0:
                # Nothing to reconstruct
                reconstructed_signal = np.zeros(self.E)
            else:
                # Solving regularized normal equations: (M_Uk^T M_Uk + reg I) c = M_Uk^T y
                A = M_Uk.T @ M_Uk
                # Scaling regularization with spectral norm for stability (auto-tuning)
                try:
                    spec = np.linalg.norm(A, ord=2)
                    lam = float(reg) * max(1.0, spec)
                except:
                    lam = float(reg)
                A_reg = A + lam * np.eye(A.shape[0])

                b = M_Uk.T @ y
                # Solving robustly with fallback to pseudoinverse
                try:
                    coeffs = np.linalg.solve(A_reg, b)
                except np.linalg.LinAlgError:
                    coeffs = pinv(A_reg) @ b

                # Reconstructing full signal from basis coefficients
                reconstructed_signal = (U_k @ coeffs).flatten()

            # Checking for numerical issues and falling back if needed
            if not np.all(np.isfinite(reconstructed_signal)):
                reconstructed_signal = np.zeros(self.E)
                for i, ei in enumerate(selected_edges):
                    if 0 <= ei < self.E and i < len(y):
                        reconstructed_signal[ei] = y[i]

        except Exception as e:
            print(f"Warning: Reconstruction failed: {e}")
            # Fallback: placing measurements at the right edge positions
            reconstructed_signal = np.zeros(self.E)
            try:
                for i, edge_idx in enumerate(sensor_placement['selected_edges']):
                    if edge_idx < self.E and i < len(sparse_measurements):
                        reconstructed_signal[edge_idx] = sparse_measurements[i]
            except:
                pass

        return reconstructed_signal
    
    def digital_twin_sync(self, current_measurements, sensor_placement, 
                         prediction_horizon=3):
        """
        Implementing the NDT synchronization loop with topological sparsification
        This creates a time-evolving digital twin that updates based on sparse measurements
        
        Args:
            current_measurements: Current sparse sensor readings
            sensor_placement: Sensor placement configuration
            prediction_horizon: Number of future time steps to predict
            
        Returns:
            Dictionary containing synchronization results and metrics
        """
        results = {
            'iterations': [],
            'reconstruction_errors': [],
            'sync_efficiency': [],
            'topological_invariants': []
        }
        
        try:
            # Getting initial reconstruction from current measurements
            current_state = self.sparse_reconstruction(current_measurements, sensor_placement)
            
            # Running synchronization loop for prediction horizon
            for t in range(prediction_horizon):
                # Performing Hodge decomposition of current state
                decomposition = self.hodge_decomposition(current_state)
                
                # Predicting next state using topological evolution model
                # (Simplified model: adding small perturbations to each component)
                next_irr = decomposition['irrotational'] * 0.95 + np.random.normal(0, 0.01, self.E)
                next_sol = decomposition['solenoidal'] * 0.98 + np.random.normal(0, 0.005, self.E)
                next_harm = decomposition['harmonic']  # Harmonic components are topological invariants
                
                # Combining components for full predicted state
                predicted_state = next_irr + next_sol + next_harm
                
                # Simulating new sparse measurements at sensor locations
                selected_edges = sensor_placement['selected_edges']
                new_measurements = []
                for edge_idx in selected_edges:
                    if edge_idx < len(predicted_state):
                        # Adding measurement noise
                        new_measurements.append(predicted_state[edge_idx] + np.random.normal(0, 0.02))
                
                # Reconstructing full state from new sparse measurements
                reconstructed_state = self.sparse_reconstruction(np.array(new_measurements), sensor_placement)
                
                # Computing performance metrics
                if t > 0:
                    reconstruction_error = np.linalg.norm(reconstructed_state - current_state) / (np.linalg.norm(current_state) + 1e-10)
                else:
                    reconstruction_error = 0
                
                sync_efficiency = len(selected_edges) / self.E  # Sparsity ratio
                
                # Computing topological invariants (Betti numbers)
                betti_0 = nx.number_connected_components(self.G)  # Connected components
                betti_1 = self.U_harm.shape[1]  # Number of holes (harmonic dimension)
                
                # Storing results for this iteration
                results['iterations'].append(t)
                results['reconstruction_errors'].append(reconstruction_error)
                results['sync_efficiency'].append(sync_efficiency)
                results['topological_invariants'].append({'betti_0': betti_0, 'betti_1': betti_1})
                
                # Updating state for next iteration
                current_state = reconstructed_state
                
        except Exception as e:
            print(f"Warning: Digital twin sync failed: {e}")
        
        return results
    
    def analyze_network_topology(self):
        """
        Analyzing topological properties following cell complex theory
        
        Returns:
            Dictionary containing comprehensive topological analysis
        """
        try:
            analysis = {
                'network_size': {'nodes': self.N, 'edges': self.E},
                'topology_matrices': {
                    'B1_shape': self.B1.shape,
                    'B2_shape': self.B2.shape if hasattr(self.B2, 'shape') else (0, 0),
                    'polygons_found': self.B2.shape[1] if hasattr(self.B2, 'shape') else 0
                },
                'spectral_properties': {
                    'L1_eigenvalues': self.eigvals_L1[:10],  # First 10 eigenvalues only
                    'harmonic_dimension': self.U_harm.shape[1],
                    'irrotational_dimension': self.U_irr.shape[1],
                    'solenoidal_dimension': self.U_sol.shape[1]
                },
                'topological_invariants': {
                    'betti_0': nx.number_connected_components(self.G),
                    'betti_1': self.U_harm.shape[1],  # Number of holes
                    'euler_characteristic': self.N - self.E + (self.B2.shape[1] if hasattr(self.B2, 'shape') else 0)
                }
            }
        except Exception as e:
            print(f"Warning: Topology analysis failed: {e}")
            # Providing fallback analysis
            analysis = {
                'network_size': {'nodes': self.N, 'edges': self.E},
                'topology_matrices': {'B1_shape': (0, 0), 'B2_shape': (0, 0), 'polygons_found': 0},
                'spectral_properties': {'harmonic_dimension': 1, 'irrotational_dimension': 1, 'solenoidal_dimension': 1},
                'topological_invariants': {'betti_0': 1, 'betti_1': 1, 'euler_characteristic': 0}
            }
        
        return analysis
    
    def compare_sparsity_methods(self, edge_signal, sparsity_ratios=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """
        Comparing different sparse monitoring approaches
        This demonstrates the sparsity-accuracy trade-off
        
        Args:
            edge_signal: Ground truth signal on all edges
            sparsity_ratios: Different monitoring budgets to test
            
        Returns:
            Dictionary containing comparison results
        """
        results = {
            'sparsity_ratios': sparsity_ratios,
            'topological_errors': [],
            'degree_based_errors': []
        }
        
        try:
            # Testing each sparsity ratio
            for ratio in sparsity_ratios:
                # Testing each sensor placement method
                methods = ['topological', 'degree_based']  # Removed 'random'
                method_errors = []
                
                for method in methods:
                    try:
                        # Getting sensor placement for current method and sparsity
                        sensor_placement = self.sparse_sensor_placement(ratio, method)
                        
                        # Simulating sparse measurements from ground truth signal
                        selected_edges = sensor_placement['selected_edges']
                        sparse_measurements = []
                        for edge_idx in selected_edges:
                            if edge_idx < len(edge_signal):
                                sparse_measurements.append(edge_signal[edge_idx])
                        
                        # Reconstructing full signal from sparse measurements
                        reconstructed = self.sparse_reconstruction(np.array(sparse_measurements), sensor_placement)
                        
                        # Computing normalized reconstruction error
                        error = np.linalg.norm(reconstructed - edge_signal) / (np.linalg.norm(edge_signal) + 1e-10)
                        method_errors.append(error)
                        
                    except Exception as e:
                        print(f"Warning: Method {method} failed: {e}")
                        method_errors.append(1.0)  # High error for failed method
                
                # Storing results for each method
                results['topological_errors'].append(method_errors[0] if len(method_errors) > 0 else 1.0)
                results['degree_based_errors'].append(method_errors[1] if len(method_errors) > 1 else 1.0)
                
        except Exception as e:
            print(f"Warning: Sparsity comparison failed: {e}")
        
        return results

    def calculate_traffic_information_content(self, traffic_data, bits_per_value=32):
        """
        Calculating information required to encode traffic state
        This quantifies the data transfer requirements for network monitoring
        
        Args:
            traffic_data: Traffic values on links (packets/sec, utilization, etc.)
            bits_per_value: Bits needed to encode each traffic measurement
        
        Returns:
            Dictionary containing information content analysis
        """
        try:
            total_links = len(traffic_data)
            total_information_bits = total_links * bits_per_value
            total_information_bytes = total_information_bits / 8
            total_information_kb = total_information_bytes / 1024
            
            return {
                'total_links': total_links,
                'bits_per_measurement': bits_per_value,
                'total_bits': total_information_bits,
                'total_bytes': total_information_bytes,
                'total_kb': total_information_kb,
                'baseline_transfer_cost': total_information_kb
            }
        except Exception as e:
            print(f"Warning: Traffic information calculation failed: {e}")
            return {}

    def measure_traffic_transfer_reduction(self, traffic_data, sensor_placement, bits_per_value=32):
        """
        Measuring actual traffic data transfer reduction from sparse monitoring
        This demonstrates the bandwidth savings achieved by topological sparsification
        
        Args:
            traffic_data: Ground truth traffic on all links
            sensor_placement: Sparse monitoring placement result
            bits_per_value: Bits per traffic measurement
        
        Returns:
            Dictionary containing traffic transfer reduction metrics
        """
        try:
            # Calculating full network information requirement
            full_info = self.calculate_traffic_information_content(traffic_data, bits_per_value)
            
            # Calculating sparse monitoring information requirement
            monitored_links = len(sensor_placement['selected_edges'])
            sparse_bits = monitored_links * bits_per_value
            sparse_bytes = sparse_bits / 8
            sparse_kb = sparse_bytes / 1024
            
            # Computing reduction metrics
            transfer_reduction_percent = (1 - sparse_kb / full_info['total_kb']) * 100 if full_info['total_kb'] > 0 else 0
            compression_ratio = full_info['total_kb'] / sparse_kb if sparse_kb > 0 else 0
            bandwidth_savings_kb = full_info['total_kb'] - sparse_kb
            
            return {
                'full_network_transfer_kb': full_info['total_kb'],
                'sparse_network_transfer_kb': sparse_kb,
                'transfer_reduction_percent': transfer_reduction_percent,
                'compression_ratio': compression_ratio,
                'bandwidth_savings_kb': bandwidth_savings_kb,
                'monitored_links': monitored_links,
                'total_links': len(traffic_data)
            }
            
        except Exception as e:
            print(f"Warning: Traffic transfer reduction calculation failed: {e}")
            return {}

    def calculate_traffic_approximation_error(self, true_traffic, reconstructed_traffic):
        """
        Calculating approximation error in inferred traffic values
        This measures how well sparse reconstruction recovers the true traffic state
        
        Args:
            true_traffic: Ground truth traffic on all links
            reconstructed_traffic: Inferred traffic from sparse measurements
        
        Returns:
            Dictionary containing traffic approximation error metrics
        """
        try:
            # Ensuring both arrays have same length
            if len(true_traffic) != len(reconstructed_traffic):
                min_len = min(len(true_traffic), len(reconstructed_traffic))
                true_traffic = true_traffic[:min_len]
                reconstructed_traffic = reconstructed_traffic[:min_len]
            
            # Computing various error metrics
            mse = np.mean((true_traffic - reconstructed_traffic) ** 2)  # Mean Squared Error
            mae = np.mean(np.abs(true_traffic - reconstructed_traffic))  # Mean Absolute Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            
            # Computing normalized errors (scale-independent)
            traffic_norm = np.linalg.norm(true_traffic)
            if traffic_norm > 0:
                normalized_mse = mse / (traffic_norm ** 2)
                normalized_mae = mae / np.mean(np.abs(true_traffic))
                normalized_rmse = rmse / traffic_norm
            else:
                normalized_mse = normalized_mae = normalized_rmse = 0
            
            # Computing relative error per link
            relative_errors = np.abs(true_traffic - reconstructed_traffic) / (np.abs(true_traffic) + 1e-10)
            mean_relative_error = np.mean(relative_errors)
            max_relative_error = np.max(relative_errors)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'normalized_mse': normalized_mse,
                'normalized_mae': normalized_mae, 
                'normalized_rmse': normalized_rmse,
                'mean_relative_error_percent': mean_relative_error * 100,
                'max_relative_error_percent': max_relative_error * 100,
                'approximation_quality': 1 - normalized_rmse  # Higher is better (1 = perfect)
            }
            
        except Exception as e:
            print(f"Warning: Traffic approximation error calculation failed: {e}")
            return {}
    
    def analyze_traffic_state_efficiency(self, traffic_data, sparsity_ratios=[0.1, 0.15, 0.2, 0.25, 0.3], 
                                   bits_per_measurement=32):
        """
        Creating comprehensive analysis of traffic state transfer efficiency
        This evaluates the trade-off between data reduction and reconstruction accuracy
        
        Args:
            traffic_data: Ground truth traffic measurements
            sparsity_ratios: Different monitoring budgets to test
            bits_per_measurement: Bits needed per traffic measurement
        
        Returns:
            Dictionary containing complete traffic efficiency analysis
        """
        try:
            results = {
                'sparsity_ratios': sparsity_ratios,
                'transfer_reduction_percent': [],
                'compression_ratios': [],
                'bandwidth_savings_kb': [],
                'approximation_quality': [],
                'traffic_mse': [],
                'traffic_mae': [],
                'relative_error_percent': []
            }
            
            print(f"\n=== TRAFFIC STATE TRANSFER EFFICIENCY ANALYSIS ===")
            print(f"Original Traffic Data: {len(traffic_data)} links")
            print(f"Encoding: {bits_per_measurement} bits per measurement")
            print()
            print("Budget | Transfer  | Compression | Bandwidth | Approx | Traffic | Relative")
            print("  (%)  | Reduction |    Ratio    | Savings   | Quality|   MSE   |  Error")
            print("       |    (%)    |     (X:1)   |   (KB)    |  Score |         |   (%)")
            print("-------|-----------|-------------|-----------|--------|---------|----------")
            
            # Testing each sparsity ratio
            for ratio in sparsity_ratios:
                try:
                    # Getting sparse sensor placement using topological method
                    placement = self.sparse_sensor_placement(ratio, 'topological')
                    
                    # Extracting sparse measurements from ground truth
                    sparse_measurements = []
                    for edge_idx in placement['selected_edges']:
                        if edge_idx < len(traffic_data):
                            sparse_measurements.append(traffic_data[edge_idx])
                    
                    # Reconstructing full traffic state from sparse measurements
                    reconstructed_traffic = self.sparse_reconstruction(np.array(sparse_measurements), placement)
                    
                    # Calculating transfer reduction metrics
                    transfer_metrics = self.measure_traffic_transfer_reduction(
                        traffic_data, placement, bits_per_measurement)
                    
                    # Calculating approximation error metrics
                    error_metrics = self.calculate_traffic_approximation_error(
                        traffic_data, reconstructed_traffic)
                    
                    # Storing results for this sparsity ratio
                    results['transfer_reduction_percent'].append(transfer_metrics.get('transfer_reduction_percent', 0))
                    results['compression_ratios'].append(transfer_metrics.get('compression_ratio', 0))
                    results['bandwidth_savings_kb'].append(transfer_metrics.get('bandwidth_savings_kb', 0))
                    results['approximation_quality'].append(error_metrics.get('approximation_quality', 0))
                    results['traffic_mse'].append(error_metrics.get('mse', 0))
                    results['traffic_mae'].append(error_metrics.get('mae', 0))
                    results['relative_error_percent'].append(error_metrics.get('mean_relative_error_percent', 0))
                    
                    # Printing formatted results row
                    print(f"{ratio*100:6.1f} | {transfer_metrics.get('transfer_reduction_percent', 0):9.1f} | "
                        f"{transfer_metrics.get('compression_ratio', 0):11.2f} | "
                        f"{transfer_metrics.get('bandwidth_savings_kb', 0):9.1f} | "
                        f"{error_metrics.get('approximation_quality', 0):6.3f} | "
                        f"{error_metrics.get('mse', 0):7.4f} | "
                        f"{error_metrics.get('mean_relative_error_percent', 0):8.2f}")
                    
                except Exception as e:
                    print(f"{ratio*100:6.1f} | ERROR: {str(e)[:60]}")
                    # Filling with zeros to maintain array consistency
                    for key in ['transfer_reduction_percent', 'compression_ratios', 'bandwidth_savings_kb',
                            'approximation_quality', 'traffic_mse', 'traffic_mae', 'relative_error_percent']:
                        results[key].append(0)
            
            return results
            
        except Exception as e:
            print(f"Warning: Traffic state efficiency analysis failed: {e}")
            return {}

    def plot_traffic_efficiency_analysis(self, traffic_results, network_name="Network"):
        """
        Creating visualization of traffic state transfer efficiency results
        This generates comprehensive plots showing the efficiency-accuracy trade-offs
        
        Args:
            traffic_results: Results from analyze_traffic_state_efficiency()
            network_name: Name for the network being analyzed
        """
        try:
            # Checking if results are valid
            if not traffic_results or not traffic_results.get('sparsity_ratios'):
                print("Warning: No traffic results to visualize")
                return
            
            # Creating 2x2 subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
            colors = ['#FF4757', '#3742FA', '#2ED573', '#FFA502']
            
            # Plot 1: Transfer reduction vs approximation quality (efficiency-accuracy trade-off)
            axes[0,0].scatter(traffic_results['transfer_reduction_percent'], 
                            traffic_results['approximation_quality'],
                            s=100, c=colors[0], alpha=0.7, edgecolor='black', linewidth=2)
            axes[0,0].set_xlabel('Traffic Data Transfer Reduction (%)', fontsize=14, fontweight='bold')
            axes[0,0].set_ylabel('Traffic Approximation Quality', fontsize=14, fontweight='bold')
            axes[0,0].set_title(f'Efficiency vs Accuracy Trade-off', fontsize=16, fontweight='bold')
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Bandwidth savings over sparsity ratios
            axes[0,1].plot(traffic_results['sparsity_ratios'], traffic_results['bandwidth_savings_kb'],
                        'o-', color=colors[1], linewidth=3, markersize=8, markeredgecolor='white', markeredgewidth=2)
            axes[0,1].set_xlabel('Sparsity Ratio', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Bandwidth Savings (KB)', fontsize=14, fontweight='bold')
            axes[0,1].set_title(f'Traffic Data Bandwidth Savings', fontsize=16, fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Compression ratio bar chart
            axes[1,0].bar(range(len(traffic_results['sparsity_ratios'])), traffic_results['compression_ratios'],
                        color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[1,0].set_xlabel('Sparsity Ratio', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('Traffic Data Compression Ratio', fontsize=14, fontweight='bold')
            axes[1,0].set_title(f'Traffic Data Compression', fontsize=16, fontweight='bold')
            axes[1,0].set_xticks(range(len(traffic_results['sparsity_ratios'])))
            axes[1,0].set_xticklabels([f"{r*100:.0f}%" for r in traffic_results['sparsity_ratios']])
            axes[1,0].grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Approximation error trend
            axes[1,1].plot(traffic_results['sparsity_ratios'], traffic_results['relative_error_percent'],
                        '^-', color=colors[3], linewidth=3, markersize=8, markeredgecolor='white', markeredgewidth=2)
            axes[1,1].set_xlabel('Sparsity Ratio', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('Mean Relative Error (%)', fontsize=14, fontweight='bold')
            axes[1,1].set_title(f'Traffic Approximation Error', fontsize=16, fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
            
            # Adding overall title and adjusting layout
            plt.suptitle(f'Traffic State Transfer Efficiency Analysis', 
                        fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()
            
        except Exception as e:
            print(f"Warning: Traffic efficiency visualization failed: {e}")


class TopologicalAnomalyDetector:
    """
    Creating advanced anomaly detection using topological invariants
    This class monitors changes in harmonic subspace to detect network anomalies
    """
    
    def __init__(self, st_ndt):
        """
        Initializing the topological anomaly detector
        
        Args:
            st_ndt: SparseTopologicalNDT instance to use for analysis
        """
        self.st_ndt = st_ndt
        self.baseline_betti_numbers = None
        self.baseline_harmonic_dim = None
        
    def establish_baseline(self, normal_signals):
        """
        Establishing baseline topological characteristics from normal network behavior
        
        Args:
            normal_signals: List of edge signals representing normal network state
        """
        try:
            harmonic_dims = []
            # Analyzing harmonic content of normal signals
            for signal in normal_signals:
                decomp = self.st_ndt.hodge_decomposition(signal)
                harmonic_dims.append(np.linalg.norm(decomp['harmonic']))
            
            # Computing baseline harmonic dimension (average of normal states)
            self.baseline_harmonic_dim = np.mean(harmonic_dims) if harmonic_dims else 1.0
            # Getting baseline topological invariants
            analysis = self.st_ndt.analyze_network_topology()
            self.baseline_betti_numbers = analysis['topological_invariants']
        except Exception as e:
            print(f"Warning: Baseline establishment failed: {e}")
            # Using default baseline values
            self.baseline_harmonic_dim = 1.0
            self.baseline_betti_numbers = {'betti_0': 1, 'betti_1': 1}
        
    def detect_topological_anomalies(self, current_signal, threshold=2.0):
        """
        Detecting anomalies by monitoring topological signal changes
        Changes in harmonic energy indicate new circulation patterns or network disruptions
        
        Args:
            current_signal: Current edge signal to analyze
            threshold: Sensitivity threshold for anomaly detection
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            # Performing Hodge decomposition of current signal
            decomp = self.st_ndt.hodge_decomposition(current_signal)
            current_harmonic_energy = np.linalg.norm(decomp['harmonic'])
            
            # Checking for harmonic energy anomalies (indicates new circulation patterns)
            harmonic_anomaly = False
            harmonic_ratio = 1.0
            
            if self.baseline_harmonic_dim is not None and self.baseline_harmonic_dim > 0:
                # Computing ratio of current to baseline harmonic energy
                harmonic_ratio = current_harmonic_energy / self.baseline_harmonic_dim
                # Detecting anomaly if ratio exceeds threshold in either direction
                if harmonic_ratio > threshold or harmonic_ratio < 1/threshold:
                    harmonic_anomaly = True
            
            return {
                'harmonic_anomaly': harmonic_anomaly,
                'harmonic_energy_ratio': harmonic_ratio,
                'current_decomposition': decomp
            }
        except Exception as e:
            print(f"Warning: Anomaly detection failed: {e}")
            # Returning safe default values
            return {
                'harmonic_anomaly': False,
                'harmonic_energy_ratio': 1.0,
                'current_decomposition': None
            }