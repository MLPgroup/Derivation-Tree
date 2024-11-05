import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import article_parser

# Load JSON from outputs folder
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['Results']

def create_graph(adj_list):
    graph = nx.DiGraph()  # Use DiGraph for directed graphs
    for node, edges in adj_list.items():
        for edge in edges:
            if edge is not None:
                graph.add_edge(node, edge)
    return graph

def plot_and_save_graphs(graph1, graph2, article_id, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    min_edge_length = 100
    # Generate a common layout based on graph1 (or a union of nodes from both graphs)
    all_nodes = set(graph1.nodes).union(set(graph2.nodes))
    common_layout = nx.spring_layout(graph1.subgraph(all_nodes), scale=min_edge_length * len(all_nodes))

    # Plot the first graph
    # pos1 = nx.spring_layout(graph1)
    # pos1 = nx.spring_layout(graph1, scale=min_edge_length * len(graph1.nodes))
    nx.draw(graph1, pos=common_layout, ax=axs[0], with_labels=True, node_color="lightblue", edge_color="gray")
    axs[0].set_title(f"Original Graph for Article {article_id}")

    # Plot the second graph
    # pos2 = nx.spring_layout(graph2, scale=min_edge_length * len(graph2.nodes))
    nx.draw(graph2, pos=common_layout, ax=axs[1], with_labels=True, node_color="lightgreen", edge_color="gray")
    axs[1].set_title(f"Comparison Graph for Article {article_id}")

    # Save the figure
    output_file = output_dir / f"article_{article_id}_comparison.png"
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Saved comparison for Article {article_id} to {output_file}")

def compare(file2, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_original = article_parser.get_manually_parsed_articles()
    data_comparison = load_json(file2)

    for cur_article_id in data_original:
        cur_original_adj_list = data_original[cur_article_id]['Adjacency List']
        
        # Find matching article in the second file
        cur_compare_article = data_comparison[f'Article ID: {cur_article_id}'] if data_comparison[f'Article ID: {cur_article_id}'] else None
        if cur_compare_article:
            cur_compare_adj_list = cur_compare_article['Adjacency List']
            
            # Create graphs for both adjacency lists
            original_graph = create_graph(cur_original_adj_list)
            compare_graph = create_graph(cur_compare_adj_list)

            # Plot and save the graphs side by side
            plot_and_save_graphs(original_graph, compare_graph, cur_article_id, output_dir)
        else:
            print(f"Article {cur_article_id} not found in both files.")
        return


if __name__ == '__main__':
    # Path to the JSON file being compare to
    comparison_file = "./outputs/Gemini/gemini_2024-09-17_10-48-52_UTC.json"
    # Output directory
    output_dir = "./outputs/Comparison_Graphs"
    compare(comparison_file, output_dir)
