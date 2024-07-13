'''
Description: Python code to output the results of algorithms
Author: Vishesh Prasad
Modification Log:
    July 10, 2024: created file and wrote working algorithm
'''

# Import modules
import os
import json

# Path to output folder
OUTPUT_FOLDER_PATH = 'outputs'

def save_important_equation_results(name, article_ids, predicted_equations, labeled_equations, algo_accuracy, algo_precision, algo_recall, algo_f1_score, algo_num_articles_used):
    # Check output folder existence
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        raise FileNotFoundError(f"The output folder with path {OUTPUT_FOLDER_PATH} was not found,")

    # Output file path
    output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f'{name}.json')

    # Clear output file
    open(output_file_path, 'w').close()

    # Output data format
    important_equation_data = {
        f"Article ID: {cur_article_id}": {
            "Labeled Equations": cur_labeled_equations,
            "Algorithm Predicted Equations": cur_predicted_equations if isinstance(cur_predicted_equations, list) else [cur_predicted_equations]
        } for cur_article_id, cur_predicted_equations, cur_labeled_equations in zip(article_ids, predicted_equations, labeled_equations)
    }
    important_equation_correctness = {
        "Number of articles used": algo_num_articles_used,
        "Accuracy": algo_accuracy,
        "Precision": algo_precision,
        "Recall": algo_recall,
        "F1 Score": algo_f1_score
    }

    # Write to data to file
    try: 
        with open(output_file_path, 'w') as json_file:
            json.dump({"Results": important_equation_data, "Correctness": important_equation_correctness}, json_file, indent=4)
        print(f"Successfully wrote outputs to {output_file_path}")
    except Exception as e:
        raise IOError(f"Failed to write to {output_file_path}: {e}")
    
