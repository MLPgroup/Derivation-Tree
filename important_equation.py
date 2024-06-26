'''
Description: Python code to parse json file and store manually parsed articles
Author: Vishesh Prasad
Modification Log:
    October 22, 2023: created file and wrote working algorithm
    January 5, 2024: accuracy script added
    January 16, 2024: edge case handling added for when no important equation found and accuracy script for input checking
    February 10, 2024: modify accuracy script to handle lists for the algo labeled equations
'''

# Import modules
import copy
import math
from collections import deque 
import article_parser


"""
get_most_important_equation(article)
Input: article -- dictionary with following values:
                    - Article ID: (string)
                    - Equation ID: (list of strings)
                    - Adjacency List (outgoing edges for each node, dict with key = string and value = list of strings)
       ret_list -- boolean which indicates to return either a list of equations or a single equation
Return: string -- most important equation in the article
Function: run a custom algorithm on the article to find the most important equation
Note: (1) The following algorithm works on a directed, acyclic graph
      (2) Currently only one equation is returned, see comment below to return > 1 equations
"""
def get_most_important_equation(article, ret_list):
    # Get required information
    article_id = article['Article ID']
    equation_list = article['Equation ID']
    adjacency_list = article['Adjacency List']

    # Bias for when number of outgoing edges are 0
    num_outgoing_bias = 1

    # Number of outgoing edges for each node
    num_outgoing = {}
    # Start nodes for algorithm, nodes for which there are no incoming edges and 
    start_nodes = set(copy.deepcopy(equation_list))

    # Iterate through all nodes
    for equation in equation_list:
        # Get number of outgoing edges for each node
        num_outgoing[equation] = len(adjacency_list[equation])
        # Add bias to number of outgoing edges
        num_outgoing[equation] += (num_outgoing_bias if adjacency_list[equation][0] != None else 0)

        # Get starting nodes:
        # Filter for no outgoing edges
        if adjacency_list[equation][0] == None:
            if equation in start_nodes:
                start_nodes.remove(equation)
        # Filter for no incoming edge
        for child_node in adjacency_list[equation]:
            if child_node in start_nodes:
                start_nodes.remove(child_node)

    # Start node manipulation for consistency
    start_nodes = list(start_nodes)
    start_nodes.sort()

    """
    Weighting system for algorithm =>
    Repeated for all possible starting nodes:
        - node_i_weight = (number of outgoing edges for node_i /
                           sum of total outgoing edges for each child node of current parent node) * 
                          (weight of current parent node of node_i)
    
    """
    # Dictionary to store computed weights of each node
    node_weights = dict.fromkeys(equation_list, 0)
    # Starting weight of algorithm (start = parent node of graph's start nodes)
    starting_weight = 10 * len(equation_list)
    # Total outgoing number of edges for current set of child nodes
    tot_outgoing = 0

    # Initialize the starting weight for the starting nodes
    tot_outgoing = sum(num_outgoing[child_node] for child_node in start_nodes)
    for start_node in start_nodes:
        node_weights[start_node] = ((num_outgoing[start_node] / tot_outgoing) * starting_weight)

    # DFS from each node to distribute out the weights to sub nodes
    for start_node in start_nodes:
        # Use current start_node as starting node for DFS
        cur_node = start_node

        # Temporary weight dictionary to measure flow from current start node
        cur_node_weights = dict.fromkeys(equation_list, 0)
        cur_node_weights[start_node] = node_weights[start_node]

        # DFS set up
        queue = deque()
        queue.append(cur_node)

        # Iterative DFS
        while queue:
            cur_node = queue.popleft()

            # Flow through weights to children nodes
            if adjacency_list[cur_node][0] != None:
                node_weight = cur_node_weights[cur_node]
                tot_outgoing = sum(num_outgoing[child_node] for child_node in adjacency_list[cur_node])
                if tot_outgoing == 0:
                    continue
                # Add weights for all children
                for child_node in adjacency_list[cur_node]:
                    queue.appendleft(child_node)
                    cur_node_weights[child_node] += ((num_outgoing[child_node] / tot_outgoing) * node_weight)
                    node_weights[child_node] += cur_node_weights[child_node]

    # No equations with weights
    if not node_weights:
        return "None"

    # Return node(s) with the maximum weight as most important node
    max_weight = max(node_weights.values())
    important_nodes = []
    epsilon = 1e-09
    for cur_node, weight in node_weights.items():
        """
        Floating point equality for multiple important equations
        math.isclose() ir requires Python 3.5
        If throwing errors, math.isclose() =>
            math.isclose(a, b, rel_tol=1e-09, abs_tol=0.0)
            = abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        """
        if math.isclose(weight, max_weight, rel_tol = epsilon):
            important_nodes.append(cur_node)

    # Use for multiple important nodes: return important_nodes
    # Returns most important node (for multiple max nodes, return the node that corresponds with the equation that shows up later in the research paper)
    if ret_list == True:
        return important_nodes
    else:
        return important_nodes[-1]


"""
get_algo_correctness_list(labeled_most_important_equations, algo_most_important_equations)
Input: labeled_most_important_equations -- list with the equation ids of the most important equation where were labeled
       algo_most_important_equations -- list with the list of equation ids of the most important equation found using the algorithm
       article_id_correctness -- list with the article ids of all labeled articles
       ret_list = if the algo returned either a list of equations or a single equation
Return: accuracy, precision, recall, f1_score, articles_used
Function: calculate important accuracy metrics for the algorithm
"""
def get_algo_correctness(labeled_most_important_equations, algo_most_important_equations, article_id_correctness, ret_list):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    articles_used = []

    # Clean up for unlabeled articles
    for i, cur_most_important_equation in enumerate(labeled_most_important_equations):
        if cur_most_important_equation is not None:
            articles_used.append(article_id_correctness[i])
            if ret_list == True:
                found_flag = False
                for cur_algo_eq in algo_most_important_equations[i]:
                    if cur_algo_eq == cur_most_important_equation:
                        found_flag = True
                        true_positive += 1
                    else:
                        false_positive += 1
                if found_flag == False:
                    false_negative += 1
            else:
                if cur_most_important_equation == algo_most_important_equations[i]:
                    true_positive += 1
                else:
                    false_negative += 1
                    false_positive += 1

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) != 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score, articles_used



"""
run_important_algo()
Input: none
Return: none
Function: Find and print the most important equation for the articles listed in the articles.json file
"""
def run_important_algo():
    articles = article_parser.get_manually_parsed_articles()
    algo_most_important_equations = []
    labeled_most_important_equations = []
    article_id_correctness = []

    # Toggle for desired output of algorithm (either a list of equations or a single equation)
    ret_list = True
    
    print("*-----------------------------------------------------------*")
    print("Important Equation Algorithm Output:")
    for article_id, article in articles.items():
        cur_most_important_equations = get_most_important_equation(article, ret_list)
        print(f"Article ID: {article_id} => algo equation(s): {cur_most_important_equations} vs. labeled equation(s): {article['Most Important Equation']}")
        algo_most_important_equations.append(cur_most_important_equations)
        labeled_most_important_equations.append(article['Most Important Equation'])
        article_id_correctness.append(article_id)
    
    algo_accuracy, algo_precision, algo_recall, algo_f1_score, algo_articles_used = get_algo_correctness(labeled_most_important_equations, algo_most_important_equations, article_id_correctness, ret_list)

    print("*-----------------------------------------------------------*")
    print("Important Equation Algorithm Correctness: ")
    print(f"Articles used for correctness calculations: {algo_articles_used}")
    print(f"Number of articles used for correctness calculations: {len(algo_articles_used)}")
    print(f"Accuracy: {algo_accuracy:.8f}")
    print(f"Precision: {algo_precision:.8f}")
    print(f"Recall: {algo_recall:.8f}")
    print(f"F1 Score: {algo_f1_score:.8f}")
    print("*-----------------------------------------------------------*")



"""
Entry point for derivation_tree_algo.py
Runs run_important_algo()
"""
if __name__ == '__main__':
    run_important_algo()
