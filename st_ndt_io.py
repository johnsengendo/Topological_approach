import os
import networkx as nx

def load_topology_zoo_graph(name='Cogentco_clean'):
    """Load network topology from GML file, safely handling duplicate edges."""
    import networkx as nx
    import os

    try:
        path = f"{name}.gml"
        if not os.path.exists(path):
            print(f"ERROR: {path} not found in current directory: {os.getcwd()}")
            print("Please place the file in the same folder as this script.")
            return None

        # ✅ Force MultiGraph type to skip duplicate-edge errors
        G_multi = nx.read_gml(path, label='id')
        if not isinstance(G_multi, nx.MultiGraph):
            G_multi = nx.MultiGraph(G_multi)

        # Count duplicates
        edge_list_multi = list(G_multi.edges())
        edge_set_simple = set(tuple(sorted(e)) for e in edge_list_multi)
        duplicates = len(edge_list_multi) - len(edge_set_simple)
        if duplicates > 0:
            print(f"⚠ Found {duplicates} duplicate edges in {name}.gml. Removing them...")

        # Convert to simple Graph to remove duplicates
        G = nx.Graph(G_multi)

        # Relabel nodes to integers
        G = nx.convert_node_labels_to_integers(G, ordering='sorted')

        # Keep largest connected component
        if not nx.is_connected(G):
            print(f"⚠ {name} network has multiple components. Keeping the largest.")
            comp = max(nx.connected_components(G), key=len)
            G = G.subgraph(comp).copy()
            G = nx.convert_node_labels_to_integers(G, ordering='sorted')

        print(f"✓ Loaded '{name}' topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

    except Exception as e:
        print(f"ERROR: Failed to load {name}.gml: {e}")
        return None