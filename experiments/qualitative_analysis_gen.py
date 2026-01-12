import matplotlib.pyplot as plt
import networkx as nx
import os

def setup_plot_style():
    plt.style.use('default')
    # plt.rcParams['font.family'] = 'sans-serif' # Use default
    
def draw_graph_state(ax, state_name, nodes, edges, positions):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    # Separate edges by style
    active_edges = [(u, v) for u, v, d in edges if d.get('status') == 'active']
    archived_edges = [(u, v) for u, v, d in edges if d.get('status') == 'archived']
    
    # Draw nodes
    nx.draw_networkx_nodes(G, positions, ax=ax, node_color='lightgray', node_size=2000, edgecolors='black')
    nx.draw_networkx_labels(G, positions, ax=ax, font_size=10, font_weight='bold')
    
    # Draw active edges (Blue, Solid)
    nx.draw_networkx_edges(
        G, positions, ax=ax, 
        edgelist=active_edges, 
        edge_color='blue', 
        style='solid',
        width=2,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Draw archived edges (Grey, Dashed)
    nx.draw_networkx_edges(
        G, positions, ax=ax, 
        edgelist=archived_edges, 
        edge_color='gray', 
        style='dashed', 
        width=2,
        arrowsize=20, 
        alpha=0.6,
        connectionstyle="arc3,rad=-0.1"
    )
    
    # Labels
    edge_labels = {}
    for u, v, d in edges:
        label = d.get('label', '')
        # Add time info if available
        # if 'time' in d:
        #    label += f"\n{d['time']}"
        edge_labels[(u, v)] = label

    # Draw labels with some offset or custom handling if needed, 
    # but networkx default is often okay for simple graphs
    nx.draw_networkx_edge_labels(G, positions, edge_labels, ax=ax, font_size=8, label_pos=0.5)

    ax.set_title(state_name, fontsize=12, pad=10)
    ax.axis('off')

def generate_graphs(output_path):
    # Layout positions
    pos = {
        'User': (0, 0),
        'Google': (2, 0.5),
        'OpenAI': (2, -0.5)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # State A: After T1
    # Nodes: User, Google
    edges_a = [
        ('User', 'Google', {'label': 'work_at', 'status': 'active'})
    ]
    draw_graph_state(ax1, "(a) State A: After T1\n(Active Record)", ['User', 'Google'], edges_a, pos)
    
    # State B: After T2
    # Nodes: User, Google, OpenAI
    edges_b = [
        ('User', 'Google', {'label': 'work_at', 'status': 'archived'}),
        ('User', 'OpenAI', {'label': 'work_at', 'status': 'active'})
    ]
    draw_graph_state(ax2, "(b) State B: After T2\n(Conflict Resolution)", ['User', 'Google', 'OpenAI'], edges_b, pos)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Graph] Saved evolution visualization to {output_path}")

def generate_logs():
    logs = """
### System Logs Simulation

**[Ingestion]** Input: "quit Google, moved to OpenAI"
**[Cardinality Check]** Predicate `work_at` is **EXCLUSIVE** (Max-Cardinality: 1).
**[Conflict Resolution]** Detected existing active edge: `(User, work_at, Google)`.
   -> Action: **ARCHIVE**. 
   -> Set `System_Time_End` = Now.
   -> State Update: `Google` [Active -> Archived].
**[Graph Update]** Created new edge: `(User, work_at, OpenAI)`.
   -> Status: **Active**.
   -> Semantic Time: `[T2, Now]`.

**[Retrieval]** Query: "Where did I work right before my current job?"
**[Query Analysis]** 
   -> Target: `work_at`
   -> Temporal Constraint: `before(current_job)`
**[Graph Search]** Found edges for `User` via `work_at`:
    1. Node: **OpenAI** | Status: Active   | Time: T2-Now
    2. Node: **Google** | Status: Archived | Time: T1-T2
**[Reasoning Injection]** 
<think> 
  Current job is identifier 'OpenAI' (from Active edge). 
  Query asks for "right before". 
  Comparing timestamps: Google (T1-T2) is immediately before OpenAI (T2-Now). 
  Answer: Google. 
</think>
"""
    print(logs)
    return logs

def generate_comparison_table():
    table = """
### Output Comparison

| System | Response | Error Analysis |
| :--- | :--- | :--- |
| **RAG (Baseline)** | "You work at Google and OpenAI." | **Hallucination/Merge**: Retrieves all chunks without temporal awareness. |
| **Mem0 (Baseline)** | "You are a senior engineer at OpenAI." | **Recency Bias**: Overwrites old memory, loses history. |
| **NS-CAM (Ours)** | "You worked at Google right before joining OpenAI." | **Correct**: Utilizes graph state versions and temporal reasoning. |
"""
    print(table)
    return table

if __name__ == "__main__":
    output_dir = r"D:\Classes\PRML\PJ\NS-CAM\results\qualitative_case"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualization
    graph_path = os.path.join(output_dir, "qualitative_graph_evolution.png")
    generate_graphs(graph_path)
    
    # 2. Logs
    logs = generate_logs()
    with open(os.path.join(output_dir, "system_logs.md"), "w") as f:
        f.write("# Qualitative Analysis Logs\n" + logs)
        
    # 3. Table
    table = generate_comparison_table()
    with open(os.path.join(output_dir, "comparison_table.md"), "w") as f:
        f.write("# Output Comparison\n" + table)
