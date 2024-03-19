'''
Description: Python code to parse article html and extract equations
Author: Vishesh Prasad
Modification Log:
    February 10, 2024: create file and extract equations from html successfully 
    February 26, 2024: use the words between equations to build the derivation tree
    March 4, 2024: implement naive bayes equation similarity
'''

from bs4 import BeautifulSoup
import os
import re
import article_parser
import sys
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


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


    
        """Playground"""
        # elif len(equations) == 0 and match_2:
        #     # Extract section and equation numbers from the matched pattern
        #     section_number, equation_number = match_2.groups()
        #     equation_key = f"S{section_number}.E{equation_number}"

        #     # Create an entry in the dictionary for the equation if not present
        #     if equation_key not in equations:
        #         equations[equation_key] = {
        #             'section_number': int(section_number),
        #             'equation_number': int(equation_number),
        #             'equations': [],
        #         }

        #     # Add the equation details to the list of equations for the current key
        #     equations[equation_key]['equations'].append({
        #         'mathml': str(mathml),
        #         'equation_id': equation_id,
        #         'alttext': alttext,
        #     })

    return equations, words_between_equations



"""
combine_sub_equations(equation)
Input: equation -- one equation in the article and all of its sub equations
Return: combined_mathml -- string with combined mathml for one equation
Function: Combine mathml for equation and all sub equations to compare with other equations
"""
def combine_sub_equations(equation):
    # Combine MathMLs of all sub-equations
    combined_mathml = ''.join(sub_equation['mathml'] for sub_equation in equation['equations'])
    return combined_mathml



"""
compute_symbol_percentage(equation1, equation2)
Input: equation1 -- mathml for one equation
       equation2 -- mathml for another equation
Return: percentage_equation1_in_equation2, percentage_equation2_in_equation1 - equation similarity percentages
Function: Compute the percentages of symbols in equation1 that are found in equation2 and vice verse
"""
def compute_symbol_percentage(equation1, equation2):
    set_equation1 = set(equation1)
    set_equation2 = set(equation2)

    percentage_equation1_in_equation2 = (len(set_equation1.intersection(set_equation2)) / len(set_equation1)) * 100
    percentage_equation2_in_equation1 = (len(set_equation2.intersection(set_equation1)) / len(set_equation2)) * 100

    return percentage_equation1_in_equation2, percentage_equation2_in_equation1



"""
equation_similarity_percentages(equations)
Input: equations -- equations found in article
Return: similarity_matrix -- [i][j] = percentage of equation i that is found in equation j
        equation_order -- order of equations in matrix
Function: Find similarity percentages between all equations
"""
def equation_similarity_percentages(equations):
    # Set up similarity matrix
    num_equations = len(equations)
    similarity_matrix = [[0.0] * num_equations for _ in range(num_equations)]

    # Combine mathml
    combined_mathml = [combine_sub_equations(equations[cur_equation]) for cur_equation in equations]
    equation_order = [cur_equation for cur_equation in equations]

    # Compute similarity percentages
    for i in range(num_equations - 1):
        equation_i = combined_mathml[i]
        for j in range(i + 1, num_equations):
            equation_j = combined_mathml[j]

            # Compute percentage similar
            percentage_i_in_j, percentage_j_in_i = compute_symbol_percentage(equation_i, equation_j)

            # Store percentages in matrix
            similarity_matrix[i][j] = percentage_i_in_j
            similarity_matrix[j][i] = percentage_j_in_i

    return similarity_matrix, equation_order


"""Playground"""
# "Check if cycle formed with new add"
# def has_cycle(adj_list, visited, current, parent):
#     visited[current] = True

#     for neighbor in adj_list[current]:
#         if not visited[neighbor]:
#             if has_cycle(adj_list, visited, neighbor, current):
#                 return True
#         elif neighbor != parent:
#             return True

#     return False



"""
equation_similarity_percentages(equations)
Input: similarity_matrix -- [i][j] = percentage of equation i that is found in equation j
        equation_order -- order of equations in matrix
        similarity_threshold -- threshold of matrix to determine if two equations are similar or not
Return: equation_adjacency_list -- adjacency list computed using 
Function: Construct an adjacency list from the similarity matrix
"""
def equation_similarity_adjacency_list(similarity_matrix, equation_order, similarity_threshold):
    num_equations = len(equation_order)
    equation_adjacency_list = {equation_order[i]: [] for i in range(num_equations)}

    for i in range(num_equations - 2, -1, -1):
        for j in range(num_equations - 1, i - 1, -1):
            if similarity_matrix[i][j] > similarity_threshold and similarity_matrix[j][i] > similarity_threshold:
                if similarity_matrix[i][j] > similarity_matrix[j][i]:
                    equation_adjacency_list[equation_order[i]].append(equation_order[j])
                else:
                    equation_adjacency_list[equation_order[j]].append(equation_order[i])

    return equation_adjacency_list



"""
bayes_classifier(article_ids, articles_used, extract_equations, extracted_words_between_equations)
Input: article_ids -- dictionary with info on all articles from articles.json
       articles_used -- list of articles where equations were extracted correctly
       extracted_equations -- list of equations that were successfully extracted
       extracted_words_between_equation -- list of list of words that occur between equations
Return: true_adjacency_lists -- list of labeled adjacency lists used in the test phase of the naive bayes algorithm
        predicted_adjacency_lists -- list of predicted adjacency lists resulting from the test phase of the naive bayes algorithm
Function: Predict adjacency list using the naive bayes algorithm
"""
def bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations):
    # Initialize lists to store true and predicted adjacency lists
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    # Split the data set into test and train
    num_articles = len(articles_used)
    train_random_indices = random.sample(range(num_articles), num_articles // 2)

    # Prepare data for the naive Bayes algorithm
    training_data = []
    training_labels = []
    train_selected_articles = set()
    for i in train_random_indices:
        training_labels.append(str(article_ids[articles_used[i]]["Adjacency List"]))
        training_data.append(' '.join([combine_sub_equations(extracted_equations[i][cur_equation]) for cur_equation in extracted_equations[i]] + extracted_words_between_equations[i]))
        train_selected_articles.add(articles_used[i])
    
    # Train the naive Bayes classifier
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(training_data)
    y_train = training_labels

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict adjacency lists for the remaining articles
    for article_id in article_ids:
        if article_id not in train_selected_articles and article_id in articles_used:
            equations = extracted_equations[articles_used.index(article_id)]
            words_between_eqs = extracted_words_between_equations[articles_used.index(article_id)]

            # Convert equations dictionary to list and concatenate it with words_between_eqs
            equations_list = [equation for sublist in equations.values() for equation in sublist]
            text_data = ' '.join(equations_list + words_between_eqs)

            # Transform the data using the same vectorizer used during training
            X_test = vectorizer.transform([text_data])

            # Predict the adjacency list using the trained classifier
            predicted_adjacency_list = classifier.predict(X_test)[0]

            # Append the true and predicted adjacency lists to the result
            true_adjacency_lists.append(article_ids[article_id]["Adjacency List"])
            predicted_adjacency_lists.append(predicted_adjacency_list)

    return true_adjacency_lists, predicted_adjacency_lists


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
            """Playground"""
            # print(cur_char)
            # print(cur_key_read)
            # print(cur_value_read)
            # print(cur_key_string)
            # print(cur_value_string)
            # print(predicted_adjacency_list)
            # print(predicted_neighbors)
            raise ValueError("Unexpected character or state encountered")

    """Playground"""
    # print("start")
    # print(predicted_adjacency_list)
    # print("middle")
    # print(predicted_neighbors)
    # print("end\n")
    return predicted_neighbors


"""
evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)
Input: true_adjacency_lists -- labeled adjacency list
       predicted_adjacency_lists -- predicted adjacency list for algorithm
Return: accuracy, precision, recall, f1_score
Function: Evaluate accuracy of classification
"""
def evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    num_skipped = 0

    for true_adjacency_list, cur_predicted_adjacency_list in zip(true_adjacency_lists, predicted_adjacency_lists):
        # If predicted adjacency list is a string, then it is from the bayes implementation
        if (isinstance(cur_predicted_adjacency_list, str)):
            predicted_adjacency_list = find_equation_neighbors_str(cur_predicted_adjacency_list)
        else:
            predicted_adjacency_list = cur_predicted_adjacency_list
        
        # Skip bad parsings
        if predicted_adjacency_list is None:
            num_skipped += 1
            continue
        
        # Calculate Error
        for equation, true_neighbors in true_adjacency_list.items():
            predicted_neighbors = predicted_adjacency_list.get(equation, [])

            for neighbor in true_neighbors:
                if neighbor in predicted_neighbors:
                    true_positive += 1
                else:
                    false_negative += 1

            for neighbor in predicted_neighbors:
                if neighbor not in true_neighbors:
                    false_positive += 1
        for equation, predicted_neighbors in predicted_adjacency_list.items():
            if equation not in true_adjacency_list:
                false_positive += len(predicted_neighbors)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) != 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score, num_skipped



"""
run_equation_similarity()
Input: algorithm_option -- type of equation similarity to run
Return: none
Function: Find the equations in articles and construct a tree depending on equation similarity
"""
def run_equation_similarity(algorithm_option):
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    extracted_equations = []
    computed_similarities = []
    equation_orders = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []
    extracted_words_between_equations = []
    articles_used = []
    

    # Iterate through article IDs
    for i, (cur_article_id, cur_article) in enumerate(article_ids.items()):
        # Construct the HTML file path for the current article
        html_path = f'articles/{cur_article_id}.html'
    
        # Check if the HTML file exists
        if os.path.exists(html_path):
            # Read the content of the HTML file
            with open(f'articles/{cur_article_id}.html', 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # Extract equations from the HTML content
            equations, words_between_equations = extract_equations(html_content)

            # If extracted correctly, compute similarity
            if len(cur_article["Equation ID"]) == len(equations) and all(cur_equation in cur_article["Equation ID"] for cur_equation in equations):
                extracted_equations.append(equations)
                extracted_words_between_equations.append(words_between_equations)
                articles_used.append(cur_article_id)

                if algorithm_option == 'string':
                    computed_similarity, equation_order = equation_similarity_percentages(equations)
                    # print(cur_article_id)
                    # print(equation_order)
                    # for row in computed_similarity:
                    #     print(' '.join(f'{percentage:.2f}' for percentage in row))
                    
                    computed_adjacency_list = equation_similarity_adjacency_list(computed_similarity, equation_order, 85)
                    # print(computed_adjacency_list)

                    computed_similarities.append(computed_similarity)
                    equation_orders.append(equation_order)
                    true_adjacency_lists.append(cur_article["Adjacency List"])
                    predicted_adjacency_lists.append(computed_adjacency_list)

        else:
            # No html for article found
            print(f"HTML file {html_path} not found")

    # Run Bayes algorithm
    if algorithm_option == 'bayes':
        true_adjacency_lists, predicted_adjacency_lists = bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations)
    
    # Get accuracy numbers
    similarity_accuracy, similarity_precision, similarity_recall, similarity_f1_score, similarity_num_skipped = evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)

    print("*-----------------------------------------------------------*")
    print("Equation Similarity Algorithm Correctness: ")
    print(f"Articles used for equation similarity correctness calculations: {len(true_adjacency_lists) - similarity_num_skipped}")
    if algorithm_option == 'string':
        print(f"Method used: String Similarity")
    elif algorithm_option == 'bayes':
        print(f"Method used: Bayes Classifier")
    print(f"Accuracy: {similarity_accuracy:.8f}")
    print(f"Precision: {similarity_precision:.8f}")
    print(f"Recall: {similarity_recall:.8f}")
    print(f"F1 Score: {similarity_f1_score:.8f}")
    print("*-----------------------------------------------------------*")

    return 0


"""
Entry point for equation_similarity.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    # Read in argument for which equation similarity algorithm to run
    if len(sys.argv) != 2:
        raise ValueError("Incorrect call, Usage: python3 equation_similarity.py <algorithm>")

    algorithm_option = sys.argv[1].lower()

    if algorithm_option not in ['bayes', 'string']:
        raise ValueError("Invalid algorithm option. Choose 'bayes' or 'string'.")
    
    # Call corresponding equation similarity function
    run_equation_similarity(algorithm_option)
