'''
Description: Python code to output the results of algorithms
Author: Vishesh Prasad
Modification Log:
    July 10, 2024: created file and wrote working algorithm
'''



# Import modules
import os
import json
import pytz
import numpy as np
from datetime import datetime



# Path to output folder
OUTPUT_FOLDER_PATHS = {
    'important_equation': './outputs/Important_Equation',
    'token': './outputs/Token_Similarity',
    'bayes': './outputs/Naive_Bayes',
    'brute': './outputs/Brute_Force',
    'gemini': './outputs/Gemini',
    'llama': './outputs/Llama',
    'mistral': './outputs/Mistral',
    'qwen': './outputs/Qwen',
    'zephyr': './outputs/Zephyr',
    'falcon': './outputs/falcon',
    'chatgpt': './outputs/chatgpt'
}

# Time Zone
TIME_ZONE = 'UTC'



"""
save_important_equation_results(name, article_ids, predicted_equations, labeled_equations, algo_accuracy, algo_precision, algo_recall, algo_f1_score, algo_num_articles_used)
Input: name -- Name of the output JSON file to write to
       article_ids -- List of article ids used in the testing
       predicted_equations -- List of list of predicted important equations for each article
       labeled_equations -- List of labeled important equation for each article
       algo_accuracy -- Accuracy score resulting from the most important equation
       algo_precision -- Precision score resulting from the most important equation
       algo_recall -- Recall score resulting from the most important equation
       algo_f1_score -- F1 score resulting from the most important equation
       algo_num_articles_used -- Number of articles the important equation algorithm was run on
Return: none
Function: Output the results of the most important equation algorithm into a JSON file
"""
def save_important_equation_results(name, article_ids, predicted_equations, labeled_equations, algo_accuracy, algo_precision, algo_recall, algo_f1_score, algo_num_articles_used):
    OUTPUT_FOLDER_PATH_IMPORTANT_EQUATION = OUTPUT_FOLDER_PATHS['important_equation']
    # Check output folder existence
    if not os.path.exists(OUTPUT_FOLDER_PATH_IMPORTANT_EQUATION):
        raise FileNotFoundError(f"The output folder with path {OUTPUT_FOLDER_PATH_IMPORTANT_EQUATION} was not found,")

    # Output file path
    output_file_path = os.path.join(OUTPUT_FOLDER_PATH_IMPORTANT_EQUATION, f'{name}.json')

    # Clear output file
    open(output_file_path, 'w').close()

    # Output data format
    important_equation_correctness = {
        "Number of articles used": algo_num_articles_used,
        "Accuracy": algo_accuracy,
        "Precision": algo_precision,
        "Recall": algo_recall,
        "F1 Score": algo_f1_score
    }
    important_equation_data = {
        f"Article ID: {cur_article_id}": {
            "Labeled Equations": cur_labeled_equations,
            "Algorithm Predicted Equations": cur_predicted_equations if isinstance(cur_predicted_equations, list) else [cur_predicted_equations]
        } for cur_article_id, cur_predicted_equations, cur_labeled_equations in zip(article_ids, predicted_equations, labeled_equations)
    }

    # Write to data to file
    try: 
        with open(output_file_path, 'w') as json_file:
            json.dump({"Correctness": important_equation_correctness, "Results": important_equation_data}, json_file, indent=4)
        print(f"Successfully wrote outputs to {output_file_path}")
    except Exception as e:
        raise IOError(f"Failed to write to {output_file_path}: {e}")



"""
save_derivation_graph_results(algo_type, name, article_ids, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, similarity_num_articles_used, train_article_ids=[])
Input: algo_type -- Which algorithm to print the results for
       name -- Name of the output JSON file to write to
       article_ids -- List of article ids for which adjacency lists are outputted
       predicted_adjacency_lists -- List of predicted adjacency lists for each article
       similarity_accuracies -- List of accuracy scores for each article
       similarity_precisions -- List of precision scores for each article
       similarity_recalls -- List of recall scores for each article
       similarity_f1_scores -- List of f1 scores for each article
       overall_accuracy -- Overall accuracy score resulting from the formulated adjacency lists
       overall_precision -- Overall precision score resulting from the formulated adjacency lists
       overall_recall -- Overall recall score resulting from the formulated adjacency lists
       overall_f1_score -- Overall f1 score resulting from the formulated adjacency lists
       similarity_num_articles_used -- Number of articles the important equation algorithm was run on
       train_article_ids -- Optional list of article ids used for algorithm training
Return: none
Function: Output the results of the given algorithm into a JSON file
"""
def save_derivation_graph_results(algo_type, name, article_ids, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, similarity_num_articles_used, train_article_ids=[]):
    # Specific output file path
    cur_output_path = OUTPUT_FOLDER_PATHS[algo_type]

    # Check output folder existence
    if not os.path.exists(cur_output_path):
        raise FileNotFoundError(f"The output folder with path {cur_output_path} was not found,")

    # Add time to title for Naive Bayes
    current_time = datetime.now(pytz.timezone(TIME_ZONE))
    timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S_%Z')

    # Output file path
    output_file_path = ""
    if algo_type == 'bayes':
        output_file_path = os.path.join(cur_output_path, f'{name}_{timestamp}.json')
    elif algo_type == 'token':
        output_file_path = os.path.join(cur_output_path, f'{name}.json')
    elif algo_type == 'brute':
        output_file_path = os.path.join(cur_output_path, f'{name}.json')
    elif algo_type == 'gemini':
        output_file_path = os.path.join(cur_output_path, f'{name}_{timestamp}.json')

    # Clear output file
    open(output_file_path, 'w').close()

    # Output data format
    overall_correctness = {
        "Number of articles used": similarity_num_articles_used,
        "Overall Correctness": {
            "Overall Accuracy": overall_accuracy,
            "Overall Precision": overall_precision,
            "Overall Recall": overall_recall,
            "Overall F1 Score": overall_f1_score
        },
        "Aggregate Correctness Statistics": {
            "Accuracy": {
                "Mean": sum(similarity_accuracies) / len(similarity_accuracies),
                "Lowest": np.min(similarity_accuracies),
                "25th Quartile (Q1)": np.percentile(similarity_accuracies, 25),
                "Median": np.percentile(similarity_accuracies, 50),
                "75th Quartile (Q3)": np.percentile(similarity_accuracies, 75),
                "Highest": np.max(similarity_accuracies)
            },
            "Precision": {
                "Mean": sum(similarity_precisions) / len(similarity_precisions),
                "Lowest": np.min(similarity_precisions),
                "25th Quartile (Q1)": np.percentile(similarity_precisions, 25),
                "Median": np.percentile(similarity_precisions, 50),
                "75th Quartile (Q3)": np.percentile(similarity_precisions, 75),
                "Highest": np.max(similarity_precisions)
            },
            "Recall": {
                "Mean": sum(similarity_recalls) / len(similarity_recalls),
                "Lowest": np.min(similarity_recalls),
                "25th Quartile (Q1)": np.percentile(similarity_recalls, 25),
                "Median": np.percentile(similarity_recalls, 50),
                "75th Quartile (Q3)": np.percentile(similarity_recalls, 75),
                "Highest": np.max(similarity_recalls)
            },
            "F1 Score": {
                "Mean": sum(similarity_f1_scores) / len(similarity_f1_scores),
                "Lowest": np.min(similarity_f1_scores),
                "25th Quartile (Q1)": np.percentile(similarity_f1_scores, 25),
                "Median": np.percentile(similarity_f1_scores, 50),
                "75th Quartile (Q3)": np.percentile(similarity_f1_scores, 75),
                "Highest": np.max(similarity_f1_scores)
            }
        }
    }
    article_data = {
        f"Article ID: {cur_article_id}": {
            "Adjacency List": cur_predicted_adjacency_lists,
            "Accuracy": cur_accuracy,
            "Precision": cur_precision,
            "Recall": cur_recall,
            "F1 Score": cur_f1
        } for cur_article_id, cur_predicted_adjacency_lists, cur_accuracy, cur_precision, cur_recall, cur_f1 in zip(article_ids, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores)
    }
    training_set = {
        "Training Articles": train_article_ids
    }
    if algo_type == 'gemini':
        training_set = {
            f"Article ID: {cur_article_id}": {
                "Parsing Error": cur_parse_error,
                "Gemini Text Response": cur_text_response
            } for cur_article_id, cur_parse_error, cur_text_response in train_article_ids
        }

    # Write to data to file
    try: 
        with open(output_file_path, 'w') as json_file:
            if algo_type == 'token':
                json.dump({"Correctness": overall_correctness, "Results": article_data}, json_file, indent=4)
            elif algo_type == 'bayes':
                json.dump({"Correctness": overall_correctness, "Results": article_data, "Training": training_set}, json_file, indent=4)
            elif algo_type == 'brute':
                json.dump({"Correctness": overall_correctness, "Results": article_data}, json_file, indent=4)
            elif algo_type == 'gemini':
                json.dump({"Correctness": overall_correctness, "Results": article_data, "Errors": training_set}, json_file, indent=4)

        print(f"Successfully wrote outputs to {output_file_path}")
    except Exception as e:
        raise IOError(f"Failed to write to {output_file_path}: {e}")
