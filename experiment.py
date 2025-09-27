# Import Modules
import article_parser 

def get_edge_stats():
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    total_outgoing_edges = 0
    total_nodes = 0
    max_outgoing_edges = 0
    total_max_outgoing_edges = 0
    total_articles = len(article_ids)

    if total_articles == 0:
        print("No articles found.")
        return

    # Iterate through article IDs
    for i, (cur_article_id, cur_article) in enumerate(article_ids.items()):
        # Get required information
        article_id = cur_article['Article ID']
        equation_list = cur_article['Equation ID']
        adjacency_list = cur_article['Adjacency List']

        # Calculate statistics for the current article
        nodes = len(adjacency_list)
        total_nodes += nodes

        outgoing_edges = sum(len(edges) for edges in adjacency_list.values())
        total_outgoing_edges += outgoing_edges

        max_edges = max((len(edges) for edges in adjacency_list.values()), default=0)
        max_outgoing_edges = max(max_outgoing_edges, max_edges)
        total_max_outgoing_edges += max_edges

    # Calculate averages
    avg_total_outgoing_edges_per_article = total_outgoing_edges / total_articles
    avg_max_outgoing_edges = total_max_outgoing_edges / total_articles
    avg_outgoing_edges_per_node = total_outgoing_edges / total_nodes if total_nodes > 0 else 0

    # Print and return results
    stats = {
        "Total Articles": total_articles,
        "Total Outgoing Edges": total_outgoing_edges,
        "Total Nodes": total_nodes,
        "Max Outgoing Edges (Single Node)": max_outgoing_edges,
        "Avg Total Outgoing Edges per Article": avg_total_outgoing_edges_per_article,
        "Avg Max Outgoing Edges per Article": avg_max_outgoing_edges,
        "Avg Outgoing Edges per Node": avg_outgoing_edges_per_node,
    }

    for stat_name, stat_value in stats.items():
        print(f"{stat_name}: {stat_value}")

    return stats

if __name__ == '__main__':
    get_edge_stats()
