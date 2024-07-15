'''
Description: Python code to parse article html and extract equations
Author: Vishesh Prasad
Modification Log:
    February 10, 2024: create file and extract equations from html successfully 
    February 26, 2024: use the words between equations to build the derivation tree
    March 4, 2024: implement naive bayes equation similarity
    March 22, 2024: improve upon naive bayes
    May 26, 2024: output results to respective files
'''

# Import Modules
from bs4 import BeautifulSoup
import os
import re
import random
import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import article_parser
import results_output


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

    return equations, words_between_equations, equation_indexing



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
equation_similarity_adjacency_list(similarity_matrix, equation_order, similarity_threshold)
Input: similarity_matrix -- [i][j] = percentage of equation i that is found in equation j
        equation_order -- order of equations in matrix
        similarity_threshold -- threshold of matrix to determine if two equations are similar or not
        similarity_direction -- direction of similarity check to add edge
        similarity_strictness -- integer value (x = 0, 1, 2) to force minimum x number of similarity values to be greater than the threshold in edge determination
Return: equation_adjacency_list -- adjacency list computed using 
Function: Construct an adjacency list from the similarity matrix
"""
def equation_similarity_adjacency_list(similarity_matrix, equation_order, similarity_threshold, similarity_direction, similarity_strictness):
    num_equations = len(equation_order)
    equation_adjacency_list = {equation_order[i]: [] for i in range(num_equations)}

    for i in range(num_equations - 2, -1, -1):
        for j in range(num_equations - 1, i - 1, -1):
            match similarity_strictness:
                case 0:
                    if similarity_direction == 'greater':
                        if similarity_matrix[i][j] > similarity_matrix[j][i]:
                            equation_adjacency_list[equation_order[i]].append(equation_order[j])
                        else:
                            equation_adjacency_list[equation_order[j]].append(equation_order[i])
                    else:
                        if similarity_matrix[i][j] < similarity_matrix[j][i]:
                            equation_adjacency_list[equation_order[i]].append(equation_order[j])
                        else:
                            equation_adjacency_list[equation_order[j]].append(equation_order[i])
                
                case 1: 
                    if similarity_matrix[i][j] > similarity_threshold or similarity_matrix[j][i] > similarity_threshold:
                        if similarity_direction == 'greater':
                            if similarity_matrix[i][j] > similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])
                        else:
                            if similarity_matrix[i][j] < similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])
                case 2:
                    if similarity_matrix[i][j] > similarity_threshold and similarity_matrix[j][i] > similarity_threshold:
                        if similarity_direction == 'greater':
                            if similarity_matrix[i][j] > similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])
                        else:
                            if similarity_matrix[i][j] < similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])


    return equation_adjacency_list


"""
extract_features_and_labels(equations, words_between_equations, equation_indexing, adjacency_list)
Input: equations -- list of equations that were successfully extracted
       words_between_equations -- list of words that occur between equations
       equation_indexing -- list of equations in the order they were found from the article
       adjacency_list (optional) -- adjacency list used to extract labels
Return: features -- extracted features of equations and words between equations 
        labels -- labels of if one equation is connected to another and the direction (+1 if 'i' points to 'j', -1 if 'j' points to 'i', and 0 for no connection)
Function: Feature and label extraction for naive bayes where a feature contains all words that occur between two equations and the two equations themselves amd label specifies their connection
"""
def extract_features_and_labels(equations, words_between_equations, equation_indexing, adjacency_list=None):
    features = []
    labels = []
    for i in range(len(equation_indexing)):
        for j in range(i+1, len(equation_indexing)):
            # Feature extraction
            # Words before 1st equation
            feature_vector = words_between_equations[j] + " "
            # 1st equation
            for k in range(len(equations[equation_indexing[i]]['equations'])):
                feature_vector += equations[equation_indexing[i]]['equations'][k]['mathml'] + " " 
            # Words between the equations
            for k in range(i + 1, j):
                feature_vector += words_between_equations[k] + " "
            # 2nd equation
            for k in range(len(equations[equation_indexing[j]]['equations'])):
                feature_vector += equations[equation_indexing[j]]['equations'][k]['mathml'] + " "
            # Words after the 2nd equation
            feature_vector += words_between_equations[j + 1] if j + 1 < len(words_between_equations) else ""

            if adjacency_list is not None:
                # Label extraction
                label = 0
                if equation_indexing[j] in adjacency_list[equation_indexing[i]]:
                    label = 1
                elif equation_indexing[i] in adjacency_list[equation_indexing[j]]:
                    label = -1
                labels.append(label)
            features.append(feature_vector)

    if adjacency_list is not None:
        return features, labels
    else:
        return features



"""
bayes_classifier(article_ids, articles_used, extract_equations, extracted_words_between_equations)
Input: article_ids -- dictionary with info on all articles from articles.json
       articles_used -- list of articles where equations were extracted correctly
       extracted_equations -- list of equations that were successfully extracted
       extracted_words_between_equation -- list of list of words that occur between equations
       extracted_equation_indexing -- list of list of equations in the order they were found from the article
       bayes_training_percentage -- percentage of dataset to use for training of Naive Bayes model
Return: true_adjacency_lists -- list of labeled adjacency lists used in the test phase of the naive bayes algorithm
        predicted_adjacency_lists -- list of predicted adjacency lists resulting from the test phase of the naive bayes algorithm
        train_article_ids -- list of article ids used to train the classifier
Function: Predict adjacency list using the naive bayes algorithm
"""
def bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations, extracted_equation_indexing, bayes_training_percentage):
    # Initialize lists to store true and predicted adjacency lists
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    # Split the data set into test and train
    num_articles = len(articles_used)
    # train_random_indices = range(int(num_articles * (bayes_training_percentage * 1.0 / 100)))
    train_size = int(num_articles * (bayes_training_percentage / 100))
    train_random_indices = random.sample(range(num_articles), train_size)


    # # Prepare data for the naive Bayes algorithm
    # training_data = []
    # training_labels = []
    # train_selected_articles = set()
    # for i in train_random_indices:
    #     training_labels.append(str(article_ids[articles_used[i]]["Adjacency List"]))
    #     training_data.append(' '.join([combine_sub_equations(extracted_equations[i][cur_equation]) for cur_equation in extracted_equations[i]] + extracted_words_between_equations[i]))
    #     train_selected_articles.add(articles_used[i])
    
    # # Train the naive Bayes classifier
    # vectorizer = CountVectorizer()
    # X_train = vectorizer.fit_transform(training_data)
    # y_train = training_labels

    # classifier = MultinomialNB()
    # classifier.fit(X_train, y_train)

    # # Predict adjacency lists for the remaining articles
    # for article_id in article_ids:
    #     if article_id not in train_selected_articles and article_id in articles_used:
    #         equations = extracted_equations[articles_used.index(article_id)]
    #         words_between_eqs = extracted_words_between_equations[articles_used.index(article_id)]

    #         # Convert equations dictionary to list and concatenate it with words_between_eqs
    #         equations_list = [equation for sublist in equations.values() for equation in sublist]
    #         text_data = ' '.join(equations_list + words_between_eqs)

    #         # Transform the data using the same vectorizer used during training
    #         X_test = vectorizer.transform([text_data])

    #         # Predict the adjacency list using the trained classifier
    #         predicted_adjacency_list = classifier.predict(X_test)[0]

    #         # Append the true and predicted adjacency lists to the result
    #         true_adjacency_lists.append(article_ids[article_id]["Adjacency List"])
    #         predicted_adjacency_lists.append(predicted_adjacency_list)


    train_features = []
    train_labels = []
    train_article_ids = []

    for i in train_random_indices:
        equations = extracted_equations[i]
        words_between_eqs = extracted_words_between_equations[i]
        equation_indexing = extracted_equation_indexing[i]

        features, labels = extract_features_and_labels (equations, words_between_eqs, equation_indexing, article_ids[articles_used[i]]["Adjacency List"])

        train_features.extend(features)
        train_labels.extend(labels)

        train_article_ids.append(article_ids[articles_used[i]]["Article ID"])

    # Train the Naive Bayes classifier
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_features)
    y_train = train_labels

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict connections for the remaining articles
    for i in range(num_articles):
        if i not in train_random_indices:
            equations = extracted_equations[i]
            words_between_eqs = extracted_words_between_equations[i]
            equation_indexing = extracted_equation_indexing[i]

            features = extract_features_and_labels(equations, words_between_eqs, equation_indexing)
            X_test = vectorizer.transform(features)

            # Predict labels
            predictions = classifier.predict(X_test)
            predicted_adjacency_list = {equation_id: [] for equation_id in equation_indexing}
            predicted_index = 0
            # Extract predictions to form adjacency list
            for j in range(len(equation_indexing)):
                for k in range(j+1, len(equation_indexing)):
                    if predictions[predicted_index] == 1:
                        predicted_adjacency_list[equation_indexing[j]].append(equation_indexing[k])
                    elif predictions[predicted_index] == -1:
                        predicted_adjacency_list[equation_indexing[k]].append(equation_indexing[j])
                    predicted_index += 1

            predicted_adjacency_lists.append(predicted_adjacency_list)
            true_adjacency_lists.append(article_ids[articles_used[i]]["Adjacency List"])

    return true_adjacency_lists, predicted_adjacency_lists, train_article_ids


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
Function: Find the equations in articles and construct a tree depending on equation similarity
"""
def run_derivation_algo(algorithm_option):
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    '''HYPER-PARAMETERS'''
    # equation_similarity_threshold - threshold of matrix to determine if two equations are similar or not
    equation_similarity_threshold = 94.5
    # equation_similarity_direction - greater (>) or lesser (<) to determine which direction to add edge to adjacency list
    equation_similarity_direction = 'greater'
    # equation_similarity_direction - 0, 1, or 2 to determine minimum number of similarity values to be greater than the threshold in edge determination
    equation_similarity_strictness = 1
    # bayes_training_percentage - percentage of dataset to use for training of Naive Bayes model
    bayes_training_percentage = 95
    '''HYPER-PARAMETERS'''

    extracted_equations = []
    extracted_equation_indexing = []
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
            equations, words_between_equations, equation_indexing = extract_equations(html_content)

            # If extracted correctly, compute similarity
            if len(cur_article["Equation ID"]) == len(equations) and all(cur_equation in cur_article["Equation ID"] for cur_equation in equations):
                extracted_equations.append(equations)
                extracted_words_between_equations.append(words_between_equations)
                articles_used.append(cur_article_id)
                extracted_equation_indexing.append(equation_indexing)

                if algorithm_option == 'equation':
                    computed_similarity, equation_order = equation_similarity_percentages(equations)
                    # print(cur_article_id)
                    # print(equation_order)
                    # for row in computed_similarity:
                    #     print(' '.join(f'{percentage:.2f}' for percentage in row))
                    
                    computed_adjacency_list = equation_similarity_adjacency_list(computed_similarity, equation_order, equation_similarity_threshold, equation_similarity_direction, equation_similarity_strictness)
                    # print(computed_adjacency_list)

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
        true_adjacency_lists, predicted_adjacency_lists, train_article_ids = bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations, extracted_equation_indexing, bayes_training_percentage)
    
    # Get accuracy numbers
    similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, similarity_num_skipped = evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)

    # print("*-----------------------------------------------------------*")
    # print("Equation Similarity Algorithm Correctness: ")
    # print(f"Articles used for equation similarity correctness calculations: {len(true_adjacency_lists) - similarity_num_skipped}")
    # if algorithm_option == 'equation':
    #     print(f"Method used: equation Similarity")
    # elif algorithm_option == 'bayes':
    #     print(f"Method used: Bayes Classifier")
    # print(f"Accuracy: {similarity_accuracy:.8f}")
    # print(f"Precision: {similarity_precision:.8f}")
    # print(f"Recall: {similarity_recall:.8f}")
    # print(f"F1 Score: {similarity_f1_score:.8f}")
    # print("*-----------------------------------------------------------*")

    output_name = f"equation_similarity_{equation_similarity_strictness}_{equation_similarity_threshold}_{equation_similarity_direction}" if algorithm_option == 'equation' else f"naive_bayes_{bayes_training_percentage}"
    results_output.save_equation_results(algorithm_option, output_name, article_ids, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, len(true_adjacency_lists) - similarity_num_skipped, train_article_ids)




"""
Entry point for equation_similarity.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Algorithms to find derivation trees")
    parser.add_argument("-a", "--algorithm", required=True, choices=['bayes', 'equation'], help="Type of algorithm to compute derivation tree: ['bayes', 'equation']")
    args = parser.parse_args()
    
    # Call corresponding equation similarity function
    run_derivation_algo(args.algorithm.lower())
