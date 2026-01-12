
import os
import json
import glob
import networkx as nx
from datetime import datetime

def analyze_latest_graph():
    # 1. Find the latest experiment directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiments'))
    pattern = os.path.join(base_dir, "test_data_locomo_*")
    
    dirs = glob.glob(pattern)
    if not dirs:
        print("No experiment directories found.")
        return

    # Sort by creation time (or name similarity since we appended timestamp)
    # Since timestamp is appended, sorting by name descending should work
    latest_dir = sorted(dirs, reverse=True)[0]
    print(f"Analyzing Graph from: {latest_dir}")
    
    graph_path = os.path.join(latest_dir, "graph.json")
    if not os.path.exists(graph_path):
        print(f"No graph.json found in {latest_dir}")
        return

    # 2. Load Graph
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Use MultiDiGraph because our system allows multiple edges between nodes
            G = nx.node_link_graph(data)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # 3. Basic Stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    print("\n=== Graph Statistics ===")
    print(f"Total Nodes: {num_nodes}")
    print(f"Total Edges: {num_edges}")
    
    if num_nodes == 0:
        print("Graph is empty.")
        return

    # 4. Connectivity
    # For directed graph, we check weakly connected components (ignoring direction)
    weakly_connected = list(nx.weakly_connected_components(G))
    num_components = len(weakly_connected)
    largest_component_size = len(max(weakly_connected, key=len)) if weakly_connected else 0
    
    print(f"Connected Components (Weak): {num_components}")
    print(f"Largest Component Size: {largest_component_size} nodes")
    
    # 5. Node Degree Analysis (Edges per Node)
    print("\n=== Top Nodes by Edge Count (Degree) ===")
    # degree returns (node, count). For MultiDiGraph, it sums in+out.
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    
    for i, (node, deg) in enumerate(degrees[:20]):
        # Analyze Out vs In
        out_d = G.out_degree(node)
        in_d = G.in_degree(node)
        print(f"{i+1}. {node} : {deg} total (Out: {out_d}, In: {in_d})")

    # 6. Edge Relation Distribution
    print("\n=== Top Relations Used ===")
    from collections import Counter
    relations = [d['relation'] for u, v, k, d in G.edges(keys=True, data=True)]
    rel_counts = Counter(relations).most_common(10)
    for rel, count in rel_counts:
        print(f"  {rel}: {count}")

    # 7. Isolated Nodes Check
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    print(f"\nIsolated Nodes: {len(isolated)}")
    if isolated:
        print(f"Sample isolated: {isolated[:5]}")

if __name__ == "__main__":
    analyze_latest_graph()
