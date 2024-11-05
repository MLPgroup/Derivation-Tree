# Import modules
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import article_parser
from matplotlib.patches import FancyArrowPatch, Circle



"""
load_json(file_path)
Input: file_path -- file path for file holding current JSON output
Return: json.load(f)['Results'] -- Results section of output holding all required information
Function: Load JSON of inputted file and return results of algorithm run
"""
def load_json(file_path):
    # Load JSON from outputs folder
    with open(file_path, 'r') as f:
        return json.load(f)['Results']



"""
create_graph(adj_list)
Input: adj_list -- adjacency list for current graph
Return: graph -- networkx graph for inputted adjacency list
Function: Construct networkx graph for inputted adjacency list
"""
def create_graph(adj_list):
    # Use DiGraph for directed graphs
    graph = nx.DiGraph()

    # Add nodes from the adjacency list
    for node in adj_list.keys():
        graph.add_node(node)  # Ensure all nodes are added

    # Add edges from the adjacency list
    for node, edges in adj_list.items():
        for edge in edges:
            if edge is not None:
                graph.add_edge(node, edge)
    return graph



"""
plot_and_save_graphs(graph1, graph2, article_id, output_dir, caption)
Input: graph1 -- networkx graph for ground truth
       graph2 -- networkx graph for comparison
       article_id -- article id for current comparison
       output_dir -- directory where to place output figures
       caption -- correctness measures to add as caption to the figure
Return: none
Function: Plot the resulting graphs and output to the correct directory
"""
def plot_and_save_graphs(graph1, graph2, article_id, output_dir, caption):
    # Base figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting variables
    min_edge_length=5.0
    curvature=0.2
    font_size=6

    # Generate a common layout based on graph1 (or a union of nodes from both graphs)
    all_nodes = set(graph1.nodes).union(set(graph2.nodes))
    common_layout = nx.spring_layout(graph1.subgraph(all_nodes), scale=min_edge_length * len(all_nodes))

    # Helper function to draw edges for each graph
    def draw_curved_edges(graph, pos, ax, curvature=0.2, node_size=5000):
        for u, v in graph.edges():
            # Define circular patches for each node to pad the arrow position
            patchA = Circle(pos[u], radius=node_size, transform=ax.transData)
            patchB = Circle(pos[v], radius=node_size, transform=ax.transData)
            
            # Create the arrow patch with padding and curvature
            arrow = FancyArrowPatch(posA=pos[u], posB=pos[v], connectionstyle=f"arc3,rad={curvature}",
                                    color="gray", arrowstyle="-|>", mutation_scale=25, lw=1,
                                    patchA=patchA, patchB=patchB)
            ax.add_patch(arrow)

    # Plot the first graph
    nx.draw_networkx_nodes(graph1, pos=common_layout, ax=axs[0], node_color="lightblue", edgecolors="black")
    nx.draw_networkx_labels(graph1, pos=common_layout, ax=axs[0], font_size=font_size)
    draw_curved_edges(graph1, common_layout, axs[0], curvature=curvature)
    axs[0].set_title(f"Original Graph for Article {article_id}")

    # Plot the second graph
    nx.draw_networkx_nodes(graph2, pos=common_layout, ax=axs[1], node_color="lightgreen", edgecolors="black")
    nx.draw_networkx_labels(graph2, pos=common_layout, ax=axs[1], font_size=font_size)
    draw_curved_edges(graph2, common_layout, axs[1], curvature=curvature)
    axs[1].set_title(f"Comparison Graph for Article {article_id}")

    # Add caption
    fig.text(0.5, 0.01, caption, ha='center', fontsize=12, wrap=True)

    # Save the figure
    output_file = output_dir / f"article_{article_id}_comparison.png"
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Saved comparison for Article {article_id} to {output_file}")



"""
compare(compare_file, output_dir)
Input: compare_file -- file path that holds JSON output to compare against
       output_dir -- directory where to place output figures
Return: none
Function: Create side-by-side figures of ground truth graphs and algorithm output
"""
def compare(compare_file, output_dir):
    # Output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from ground truth and comparison JSONs
    data_original = article_parser.get_manually_parsed_articles()
    data_comparison = load_json(compare_file)

    # For each article
    for cur_article_id in data_original:
        # Get ground truth adjacency list
        cur_original_adj_list = data_original[cur_article_id]['Adjacency List']
        
        # Find matching article in the second file
        cur_compare_article = data_comparison[f'Article ID: {cur_article_id}'] if f'Article ID: {cur_article_id}' in data_comparison else None
        # If corresponding outputted adjacency list exists
        if cur_compare_article:
            # Get outputted adjacency list
            cur_compare_adj_list = cur_compare_article['Adjacency List']
            
            # Create graphs for both adjacency lists
            original_graph = create_graph(cur_original_adj_list)
            compare_graph = create_graph(cur_compare_adj_list)

            # Correctness metrics as caption for graph
            caption = f"Accuracy = {cur_compare_article['Accuracy']}, Precision = {cur_compare_article['Precision']}, Recall = {cur_compare_article['Recall']}, F1 Score = {cur_compare_article['F1 Score']}"

            # Plot and save the graphs side by side
            plot_and_save_graphs(original_graph, compare_graph, cur_article_id, output_dir, caption)
        else:
            print(f"Article {cur_article_id} not found in both files.")



"""
Entry point for graph_compare.py
Runs compare()
"""
if __name__ == '__main__':
    # Path to the JSON file being compare to
    comparison_file = "./outputs/Gemini/gemini_2024-09-17_10-48-52_UTC.json"
    # Output directory
    output_dir = "./outputs/Comparison_Graphs"
    # Run compare
    compare(comparison_file, output_dir)
