'''
Description: Python code to parse json file and store manually parsed articles
Author: Vishesh Prasad
Modification Log:
    October 22, 2023: created file and wrote working algorithm
    January 5, 2024: accuracy script added
'''

# Import modules
import copy
import math
from collections import deque 


"""
get_most_important_equation(article)
Input: article -- dictionary with following values:
                    - Article ID: (string)
                    - Equation ID: (list of strings)
                    - Adjacency List (outgoing edges for each node, dict with key = string and value = list of strings)
Return: string -- most important equation in the article
Function: run a custom algorithm on the article to find the most important equation
Note: The following algorithm works on a directed, acyclic graph
"""
def get_most_important_equation(article):
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
        # No outgoing edges
        if adjacency_list[equation][0] == None:
            if equation in start_nodes:
                start_nodes.remove(equation)
        # No incoming edge
        for outgoing_edge in adjacency_list[equation]:
            if outgoing_edge in start_nodes:
                start_nodes.remove(outgoing_edge)

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

    # BFS from each node to distribute out the weights to sub nodes
    for start_node in start_nodes:
        # Use current start_node as starting node for BFS
        cur_node = start_node

        # Temporary weight dictionary to measure flow from current start node
        cur_node_weights = dict.fromkeys(equation_list, 0)
        cur_node_weights[start_node] = node_weights[start_node]

        # BFS set up
        queue = deque()
        queue.append(cur_node)

        # Iterative BFS
        while queue:
            cur_node = queue.popleft()

            # Flow through weights to children nodes
            if adjacency_list[cur_node][0] != None:
                node_weight = cur_node_weights[cur_node]
                tot_outgoing = sum(num_outgoing[child_node] for child_node in adjacency_list[cur_node])
                # Add weights for all children
                for child_node in adjacency_list[cur_node]:
                    queue.appendleft(child_node)
                    cur_node_weights[child_node] += ((num_outgoing[child_node] / tot_outgoing) * node_weight)
                    node_weights[child_node] += cur_node_weights[child_node]

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

    return max(important_nodes)

"""
get_algo_correctness(labeled_most_important_equations, algo_most_important_equations)
Input: labeled_most_important_equations -- list with the equation ids of the most important equation where were labeled
       algo_most_important_equations -- list with the equation ids of the most important equation found using the algorithm
Return: confusion matrix, accuracy, precision, recall, specificity, f1 score
Function: calculate the confusion matrix for the algorithm
"""
def get_algo_correctness(labeled_most_important_equations, algo_most_important_equations):
    true_labels = []
    algo_labels = []
    # Clean up for unlabeled articles
    for i, cur_most_important_equation in enumerate(labeled_most_important_equations):
        if cur_most_important_equation != None:
            true_labels.append(labeled_most_important_equations[i])
            algo_labels.append(algo_most_important_equations[i])
    
    unique_labels = set(true_labels + algo_labels)
    num_classes = len(unique_labels)

    # Create a mapping from labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Fill in the confusion matrix
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(true_labels, algo_labels):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        conf_matrix[true_index][pred_index] += 1
    
    # Calculate metrics
    TP = sum(conf_matrix[i][i] for i in range(num_classes))
    TN = sum(sum(conf_matrix[i][j] for j in range(num_classes) if i != j) for i in range(num_classes))
    FP = sum(conf_matrix[i][j] for i in range(num_classes) for j in range(num_classes) if i != j)
    FN = sum(conf_matrix[i][j] for i in range(num_classes) for j in range(num_classes) if i != j)

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return conf_matrix, accuracy, precision, recall, specificity, f1_score
