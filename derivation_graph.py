'''
Description: Python code to get derivation graphs
Author: Vishesh Prasad
Modification Log:
    February 10, 2024: create file and extract equations from html successfully 
    February 26, 2024: use the words between equations to build the derivation graph
    March 4, 2024: implement naive bayes equation similarity
    March 22, 2024: improve upon naive bayes
    May 26, 2024: output results to respective files
    August 18, 2024: reformat file system
'''



# Import Modules
import os
import argparse
import article_parser
import results_output
import token_similarity
import naive_bayes
import brute_force
import gemini
import run_llms
import google.generativeai as genai
from collections import deque



'''HYPER-PARAMETERS'''
# NOTE: for all hyper-parameters ONLY INCLUDE DECIMAL IF THRESHOLD IS NOT AN INTEGER

# TOKEN_SIMILARITY_THRESHOLD - threshold of matrix to determine if two equations are similar or not
TOKEN_SIMILARITY_THRESHOLD = 90

# TOKEN_SIMILARITY_DIRECTION - greater (>) or lesser (<) to determine which direction to add edge to adjacency list
TOKEN_SIMILARITY_DIRECTION = 'greater'

# TOKEN_SIMILARITY_STRICTNESS - 0, 1, or 2 to determine minimum number of similarity values to be greater than the threshold in edge determination
TOKEN_SIMILARITY_STRICTNESS = 1
# BAYES_K_FOLDS - number of folds for k-fold cross-validation of Naive Bayes model
BAYES_K_FOLDS = 5

'''HYPER-PARAMETERS'''




"""
find_equation_neighbors_str(predicted_adjacency_list)
Input: predicted_adjacency_list -- labeled adjacency list as a string 
Return: dictionary with equations and predicted neighbors
Function: Convert the string of the predicted adjacency list from the bayes classifier into a dictionary
"""
def find_equation_neighbors_str(predicted_adjacency_list):
    predicted_neighbors = {}
    cur_key_read = False
    cur_value_read = False
    cur_value_string = ""
    cur_key_string = ""

    for cur_char in predicted_adjacency_list:
        # Ignore
        if cur_char in ["{", "}", ":", " ", ","]:
            continue
        # Start reading in key
        elif cur_char == "'" and not cur_key_read and not cur_value_read:
            cur_key_read = True
            cur_key_string = ""
        # Stop reading key
        elif cur_char == "'" and cur_key_read and not cur_value_read:
            cur_key_read = False
            predicted_neighbors[cur_key_string] = []
        # Start reading in values
        elif cur_char == "[" and not cur_value_read and not cur_key_read:
            cur_value_read = True
        # Stop reading in values
        elif cur_char == "]" and cur_value_read and not cur_key_read:
            cur_value_read = False
            cur_value_string = ""
        # Start read new value
        elif cur_char == "'" and len(cur_value_string) == 0:
            continue
        # End read new value
        elif cur_char == "'" and len(cur_value_string) != 0:
            predicted_neighbors[cur_key_string].append(cur_value_string)
            cur_value_string = ""
        # Read char of key
        elif cur_key_read and not cur_value_read:
            cur_key_string += cur_char
        # Read char of value
        elif cur_value_read and not cur_key_read:
            cur_value_string += cur_char
        # Error
        else:
            raise ValueError("Unexpected character or state encountered")

    """Playground"""
    return predicted_neighbors


def evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists):
    """
    Inputs:
        true_adjacency_lists       -- list of dicts: {node: [neighbor, ...], ...}
        predicted_adjacency_lists  -- list of dicts (or strings handled by find_equation_neighbors_str)
    Returns:
        accuracies, precisions, recalls, f1_scores,
        overall_accuracy, overall_precision, overall_recall, overall_f1_score,
        num_skipped
    Notes:
        - Edge-set evaluation: TP/FP/FN only, no TN. Matches LLM pipeline compute_article_tp_fp_fn.
        - accuracies field is kept for API compatibility but returns 0.0 (undefined without TN).
    """
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    overall_tp = overall_fp = overall_fn = 0
    num_skipped = 0

    for cur_true_adj, cur_pred_adj in zip(true_adjacency_lists, predicted_adjacency_lists):
        if isinstance(cur_pred_adj, str):
            predicted_adj = find_equation_neighbors_str(cur_pred_adj)
        else:
            predicted_adj = cur_pred_adj

        if predicted_adj is None:
            num_skipped += 1
            continue

        true_edges = {
            (src, tgt)
            for src, tgts in cur_true_adj.items()
            for tgt in (tgts or [])
            if tgt is not None
        }
        pred_edges = {
            (src, tgt)
            for src, tgts in predicted_adj.items()
            for tgt in (tgts or [])
            if tgt is not None
        }

        tp = len(true_edges & pred_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        accuracies.append(0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall    = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1_score  = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    return accuracies, precisions, recalls, f1_scores, 0.0, overall_precision, overall_recall, overall_f1_score, num_skipped




"""
run_derivation_algo(algorithm_option)
Input: algorithm_option -- type of equation similarity to run
Return: none
Function: Find the equations in articles and construct a graph depending on equation similarity
"""
def run_derivation_algo(algorithm_option):
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    # Variables to be tracked
    extracted_equations = []
    extracted_equation_indexing = []
    computed_similarities = []
    equation_orders = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []
    extracted_words_between_equations = []
    articles_used = []
    train_article_ids = []

    # Reset api tracking and setup model
    if algorithm_option in ['gemini', 'combine', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'edge_limit']:
        gemini.api_call_times = deque()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # gemini_model = genai.GenerativeModel("gemini-2.5-pro")
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    # Need more billing
    elif algorithm_option in ['chatgpt', 'combine_chatgpt', 'chatgptfewshot']:
        chatgpt_client = run_llms.configure_chatgpt(api_key=os.environ["OPENAI_API_KEY"])
    elif algorithm_option in ['llama', 'mistral', 'qwen', 'zephyr', 'phi']:
        run_llms.api_call_times_queue = deque()
        llm_client = run_llms.configure_hf(input_hf_token=os.environ["HF_TOKEN"], algorithm_option=algorithm_option)
    
    fewshot_articles = ["0907.2648", "1701.00847"]

    # Iterate through article IDs
    if algorithm_option not in ['brute', 'combine', 'combine_chatgpt']:
        for i, (cur_article_id, cur_article) in enumerate(article_ids.items()):
            # Fewshot articles only for fewshot
            if algorithm_option in ['geminifewshot', 'chatgptfewshot'] and cur_article_id in fewshot_articles:
                continue

            # Construct the HTML file path for the current article
            html_path = f'articles/{cur_article_id.replace("/", "_")}.html'
        
            # Check if the HTML file exists
            if os.path.exists(html_path):
                # Read the content of the HTML file
                with open(f'articles/{cur_article_id.replace("/", "_")}.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    
                # Extract equations from the HTML content
                equations, words_between_equations, equation_indexing = article_parser.extract_equations(html_content)

                # Filter to only equations listed in JSON (drops appendix/supplemental extras)
                expected_ids = set(cur_article["Equation ID"])
                equation_indexing = [e for e in equation_indexing if e in expected_ids]
                equations = {k: v for k, v in equations.items() if k in expected_ids}
                words_between_equations = words_between_equations[:len(equation_indexing) + 1]

                # If extracted correctly, continue
                if set(equations.keys()) == expected_ids:
                    # Save variables
                    extracted_equations.append(equations)
                    extracted_words_between_equations.append(words_between_equations)
                    extracted_equation_indexing.append(equation_indexing)

                    # Gather articles for naive bayes
                    if algorithm_option == 'bayes':
                        articles_used.append(cur_article_id)
                    # Token Similarity
                    elif algorithm_option in ['token', 'trev']:
                        # Get similarity matrices
                        computed_similarity, equation_order = token_similarity.token_similarity_percentages(equations)
                        
                        # Get resulting adjacency list
                        computed_adjacency_list = token_similarity.token_similarity_adjacency_list(computed_similarity, equation_order, TOKEN_SIMILARITY_THRESHOLD, TOKEN_SIMILARITY_DIRECTION, TOKEN_SIMILARITY_STRICTNESS)

                        # Save variables
                        computed_similarities.append(computed_similarity)
                        equation_orders.append(equation_order)
                        true_adjacency_lists.append(cur_article["Adjacency List"])
                        predicted_adjacency_lists.append(computed_adjacency_list)
                        articles_used.append(cur_article_id)
                        train_article_ids = []
                    # Gemini model
                    elif algorithm_option in ['gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'chatgpt', 'chatgptfewshot', 'edge_limit']:
                        if algorithm_option in ['gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'edge_limit']:
                            # Call Gemini API and get resulting adjacency list
                            computed_adjacency_list, error, error_string = gemini.get_gemini_adj_list(gemini_model, equations, words_between_equations, equation_indexing, True if algorithm_option == 'geminifewshot' else False, 1 if algorithm_option == 'grev1' else (2 if algorithm_option == 'grev2' else (3 if algorithm_option == 'grev3' else (4 if algorithm_option == 'edge_limit' else 0))))
                        elif algorithm_option in ['chatgpt', 'chatgptfewshot']:
                            computed_adjacency_list, error, error_string = run_llms.get_chatgpt_adj_list(chatgpt_client, equations, words_between_equations, equation_indexing, cur_article_id, True if algorithm_option == 'chatgptfewshot' else False)
                        else:
                            computed_adjacency_list, error, error_string = run_llms.get_llm_adj_list(llm_client, equations, words_between_equations, equation_indexing)

                        # No error
                        if error == 0:
                            predicted_adjacency_lists.append(computed_adjacency_list)
                            true_adjacency_lists.append(cur_article["Adjacency List"])
                            articles_used.append(cur_article_id)
                        # Response parsing error
                        elif error == -1:
                            train_article_ids.append((cur_article_id, error_string, computed_adjacency_list))
                        # Unknown error
                        elif error == 1:
                            train_article_ids.append((cur_article_id, error_string, computed_adjacency_list))
                        print(f"Article {cur_article_id} done")

            else:
                # No html for article found
                print(f"HTML file {html_path} not found")

    # Run Bayes algorithm
    if algorithm_option == 'bayes':
        true_adjacency_lists, predicted_adjacency_lists, train_article_ids = naive_bayes.bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations, extracted_equation_indexing, BAYES_K_FOLDS)
    elif algorithm_option == 'brute':
        articles_used, true_adjacency_lists, predicted_adjacency_lists = brute_force.brute_force_algo()
    elif algorithm_option in ['combine', 'combine_chatgpt']:
        # Use brute force to get explicit edges and llm to get implicit edges
        combine_articles_used, _ , combine_predicted_adjacency_lists = brute_force.brute_force_algo()
        for cur_article_id, cur_explicit_adj_list in zip(combine_articles_used, combine_predicted_adjacency_lists):
            cur_article = article_ids[cur_article_id]
            # Construct the HTML file path for the current article
            html_path = f'articles/{cur_article_id.replace("/", "_")}.html'

            # Check if the HTML file exists
            if os.path.exists(html_path):
                # Read the content of the HTML file
                with open(f'articles/{cur_article_id.replace("/", "_")}.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    
                # Extract equations from the HTML content
                equations, words_between_equations, equation_indexing = article_parser.extract_equations(html_content)

                # Filter to only equations listed in JSON (drops appendix/supplemental extras)
                expected_ids = set(cur_article["Equation ID"])
                equation_indexing = [e for e in equation_indexing if e in expected_ids]
                equations = {k: v for k, v in equations.items() if k in expected_ids}
                words_between_equations = words_between_equations[:len(equation_indexing) + 1]

                # If extracted correctly, continue
                if set(equations.keys()) == expected_ids:
                    # Save variables
                    extracted_equations.append(equations)
                    extracted_words_between_equations.append(words_between_equations)
                    extracted_equation_indexing.append(equation_indexing)

                    if algorithm_option == 'combine_chatgpt':
                        computed_adjacency_list, error, error_string = run_llms.get_chatgpt_combined_adj_list(chatgpt_client, equations, words_between_equations, equation_indexing, cur_explicit_adj_list)
                    else:   
                        computed_adjacency_list, error, error_string = gemini.get_combine_adj_list(gemini_model, equations, words_between_equations, equation_indexing, cur_explicit_adj_list)

                    # No error
                    if error == 0:
                        predicted_adjacency_lists.append(computed_adjacency_list)
                        true_adjacency_lists.append(cur_article["Adjacency List"])
                        articles_used.append(cur_article_id)
                    # Response parsing error
                    elif error == -1:
                        train_article_ids.append((cur_article_id, error_string, computed_adjacency_list))
                    # Unknown error
                    elif error == 1:
                        train_article_ids.append((cur_article_id, error_string, computed_adjacency_list))
                    print(f"Article {cur_article_id} done")
            
    
    # Get accuracy numbers
    similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, num_skipped = evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)

    # Name formatting
    if algorithm_option in ['token', 'trev']:
        output_name = f"token_similarity_{TOKEN_SIMILARITY_STRICTNESS}_{TOKEN_SIMILARITY_THRESHOLD}_{TOKEN_SIMILARITY_DIRECTION}"
    elif algorithm_option == 'bayes':
        output_name = f"naive_bayes_{BAYES_K_FOLDS}fold"
    elif algorithm_option == 'brute':
        output_name = f'brute_force'
    elif algorithm_option in ['gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'combine', 'chatgpt', 'combine_chatgpt', 'chatgptfewshot', 'edge_limit']:
        output_name = f"{algorithm_option}"

    # Save results
    results_output.save_derivation_graph_results(algorithm_option, output_name, articles_used, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, len(true_adjacency_lists) - num_skipped, train_article_ids)



"""
Entry point for derivation_graph.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Algorithms to find derivation graphs")
    parser.add_argument("-a", "--algorithm", required=True, choices=['bayes', 'token', 'trev', 'brute', 'gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'chatgpt', 'combine', 'combine_chatgpt', 'chatgptfewshot', 'edge_limit'], help="Type of algorithm to compute derivation graph: ['bayes', 'token', 'trev', 'brute', 'gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'chatgpt', 'combine', 'combine_chatgpt', 'chatgptfewshot', 'edge_limit]")
    args = parser.parse_args()
    
    # Call corresponding equation similarity function
    run_derivation_algo(args.algorithm.lower())
