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
from bs4 import BeautifulSoup
import os
import re
import argparse
import article_parser
import results_output
import token_similarity
import naive_bayes
import brute_force



'''HYPER-PARAMETERS'''
# NOTE: for all hyper-parameters ONLY INCLUDE DECIMAL IF THRESHOLD IS NOT AN INTEGER

# TOKEN_SIMILARITY_THRESHOLD - threshold of matrix to determine if two equations are similar or not
TOKEN_SIMILARITY_THRESHOLD = 80

# TOKEN_SIMILARITY_DIRECTION - greater (>) or lesser (<) to determine which direction to add edge to adjacency list
TOKEN_SIMILARITY_DIRECTION = 'greater'

# TOKEN_SIMILARITY_STRICTNESS - 0, 1, or 2 to determine minimum number of similarity values to be greater than the threshold in edge determination
TOKEN_SIMILARITY_STRICTNESS = 2
# BAYES_TRAINING_PERCENTAGE - percentage of dataset to use for training of Naive Bayes model
BAYES_TRAINING_PERCENTAGE = 80

'''HYPER-PARAMETERS'''



"""
extract_equations(html_content)
Input: html_content -- html content for current article that needs to be parsed
Return: equations -- equations that were found in the article
        words_between_equations -- words that occur between the equations in the article
Function: Find and return all the equations, their ids, equation content, and words between equations from the given article
"""
def extract_equations(html_content):
    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Dictionary to store equations
    equations = {}

    # List to store equations at each index
    equation_indexing = []
    
    # List to store words that occur between equations
    words_between_equations = []
    last_eq_id = "none"
    last_update_id = "none"
    
    # Define the pattern to match equations
    pattern = re.compile(r'S(\d+)\.E(\d+)')
    # pattern_2 = re.compile(r'S(\d+)\.Ex(\d+)')

    # Iterate through all 'math' elements in the HTML
    # for mathml in soup.find_all('math'):
    for item in soup.recursiveChildGenerator():
        if item.name == 'math':
            # Get equation ID and alt text attributes
            equation_id = item.get('id', '')
            alttext = item.get('alttext', '')

            # Check if the equation ID matches the defined pattern
            match = pattern.search(equation_id)
            # match_2 = pattern_2.search(equation_id)
            if match:
                # Extract section and equation numbers from the matched pattern
                section_number, equation_number = match.groups()
                equation_key = f"S{section_number}.E{equation_number}"
                last_eq_id = equation_id

                # Create an entry in the dictionary for the equation if not present
                if equation_key not in equations:
                    equations[equation_key] = {
                        'section_number': int(section_number),
                        'equation_number': int(equation_number),
                        'equations': [],
                    }
                    equation_indexing.append(equation_key)

                # Add the equation details to the list of equations for the current key
                equations[equation_key]['equations'].append({
                    'mathml': str(item),
                    'equation_id': equation_id,
                    'alttext': alttext,
                })

        # If string
        elif isinstance(item, str):
            # If before any equation
            if last_eq_id == "none":
                # If already found words
                if words_between_equations:
                    words_between_equations[-1] += item
                else: 
                    words_between_equations.append(item)
            else:
                # If new equation found
                if last_eq_id != last_update_id:
                    words_between_equations.append(item)
                else:
                    words_between_equations[-1] += item
            # Equation when updated
            last_update_id = last_eq_id

    return equations, words_between_equations, equation_indexing



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


"""
evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)
Input: true_adjacency_lists -- labeled adjacency list
       predicted_adjacency_lists -- predicted adjacency list for algorithm
Return: accuracy, precision, recall, and f1_score for each article tested on and the overall accuracy, precision, recall, and f1_score for the algorithm as a whole
Function: Evaluate accuracy of classification
"""
def evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    overall_true_positive = 0
    overall_true_negative = 0
    overall_false_positive = 0
    overall_false_negative = 0
    num_skipped = 0

    for true_adjacency_list, cur_predicted_adjacency_list in zip(true_adjacency_lists, predicted_adjacency_lists):
        # If predicted adjacency list is a string, then it is from the bayes implementation
        if (isinstance(cur_predicted_adjacency_list, str)):
            predicted_adjacency_list = find_equation_neighbors_str(cur_predicted_adjacency_list)
            ''' ----------- CAN GET RID OF DUE TO CHANGE -----------'''
        else:
            predicted_adjacency_list = cur_predicted_adjacency_list
        
        # Skip bad parsings
        if predicted_adjacency_list is None:
            num_skipped += 1
            continue
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        # Calculate Error
        for equation, true_neighbors in true_adjacency_list.items():
            predicted_neighbors = predicted_adjacency_list.get(equation, [])

            for neighbor in true_neighbors:
                if neighbor in predicted_neighbors:
                    true_positive += 1
                    overall_true_positive += 1
                else:
                    false_negative += 1
                    overall_false_negative += 1

            for neighbor in predicted_neighbors:
                if neighbor not in true_neighbors:
                    false_positive += 1
                    overall_false_positive += 1
        for equation, predicted_neighbors in predicted_adjacency_list.items():
            if equation not in true_adjacency_list:
                false_positive += len(predicted_neighbors)
                overall_false_positive += len(predicted_neighbors)

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) != 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    overall_accuracy = (overall_true_positive + overall_true_negative) / (overall_true_positive + overall_true_negative + overall_false_positive + overall_false_negative) if (overall_true_positive + overall_true_negative + overall_false_positive + overall_false_negative) != 0 else 0
    overall_precision = overall_true_positive / (overall_true_positive + overall_false_positive) if (overall_true_positive + overall_false_positive) != 0 else 0
    overall_recall = overall_true_positive / (overall_true_positive + overall_false_negative) if (overall_true_positive + overall_false_negative) != 0 else 0
    overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) != 0 else 0

    return accuracies, precisions, recalls, f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, num_skipped



"""
run_derivation_algo(algorithm_option)
Input: algorithm_option -- type of equation similarity to run
Return: none
Function: Find the equations in articles and construct a graph depending on equation similarity
"""
def run_derivation_algo(algorithm_option):
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    extracted_equations = []
    extracted_equation_indexing = []
    computed_similarities = []
    equation_orders = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []
    extracted_words_between_equations = []
    articles_used = []
    

    # Iterate through article IDs
    if algorithm_option != 'brute':
        for i, (cur_article_id, cur_article) in enumerate(article_ids.items()):
            # Construct the HTML file path for the current article
            html_path = f'articles/{cur_article_id}.html'
        
            # Check if the HTML file exists
            if os.path.exists(html_path):
                # Read the content of the HTML file
                with open(f'articles/{cur_article_id}.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    
                # Extract equations from the HTML content
                equations, words_between_equations, equation_indexing = extract_equations(html_content)

                # If extracted correctly, compute similarity
                if (len(cur_article["Equation ID"]) == len(equations)) and (all(cur_equation in cur_article["Equation ID"] for cur_equation in equations)):
                    extracted_equations.append(equations)
                    extracted_words_between_equations.append(words_between_equations)
                    articles_used.append(cur_article_id)
                    extracted_equation_indexing.append(equation_indexing)

                    if algorithm_option == 'token':
                        computed_similarity, equation_order = token_similarity.token_similarity_percentages(equations)
                        
                        computed_adjacency_list = token_similarity.token_similarity_adjacency_list(computed_similarity, equation_order, TOKEN_SIMILARITY_THRESHOLD, TOKEN_SIMILARITY_DIRECTION, TOKEN_SIMILARITY_STRICTNESS)

                        computed_similarities.append(computed_similarity)
                        equation_orders.append(equation_order)
                        true_adjacency_lists.append(cur_article["Adjacency List"])
                        predicted_adjacency_lists.append(computed_adjacency_list)
                        train_article_ids = []

            else:
                # No html for article found
                print(f"HTML file {html_path} not found")

    # Run Bayes algorithm
    if algorithm_option == 'bayes':
        true_adjacency_lists, predicted_adjacency_lists, train_article_ids = naive_bayes.bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations, extracted_equation_indexing, BAYES_TRAINING_PERCENTAGE)
    elif algorithm_option == 'brute':
        train_article_ids, true_adjacency_lists, predicted_adjacency_lists = brute_force.brute_force_algo()
    
    # Get accuracy numbers
    similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, num_skipped = evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)

    if algorithm_option == 'token':
        output_name = f"token_similarity_{TOKEN_SIMILARITY_STRICTNESS}_{TOKEN_SIMILARITY_THRESHOLD}_{TOKEN_SIMILARITY_DIRECTION}"
    elif algorithm_option == 'bayes':
        output_name = f"naive_bayes_{BAYES_TRAINING_PERCENTAGE}"
    elif algorithm_option == 'brute':
        output_name = f'brute_force'

    results_output.save_derivation_graph_results(algorithm_option, output_name, articles_used, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, len(true_adjacency_lists) - num_skipped, train_article_ids)




"""
Entry point for derivation_graph.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Algorithms to find derivation graphs")
    parser.add_argument("-a", "--algorithm", required=True, choices=['bayes', 'token', 'brute'], help="Type of algorithm to compute derivation graph: ['bayes', 'token', 'brute']")
    args = parser.parse_args()
    
    # Call corresponding equation similarity function
    run_derivation_algo(args.algorithm.lower())
