import os
import json
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


EQUATION_SIMILARITY_PATH = './outputs/Equation_Similarity'
NAIVE_BAYES_PATH = './outputs/Naive_Bayes'
PLOT_PATH = './outputs/plots'


# Plot equation similarity results
def plot_equation_similarity(alg_num_threshold, alg_direction):

    # Extract threshold 
    def extract_threshold(filename):
        # Extract the numeric value from the filename
        match = re.search(fr'equation_similarity_{alg_num_threshold}_(\d+\.?\d*)_{alg_direction}', filename)
        if match:
            return float(match.group(1))
        return None


    # Extract data from json files
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    thresholds = []
    data_files = []

    for filename in os.listdir(EQUATION_SIMILARITY_PATH):
        if filename.endswith('.json') and f'equation_similarity_{alg_num_threshold}_' in filename and f'{alg_direction}' in filename:
            file_path = os.path.join(EQUATION_SIMILARITY_PATH, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                accuracy = data["Correctness"]["Overall Correctness"]["Overall Accuracy"]
                precision = data["Correctness"]["Overall Correctness"]["Overall Precision"]
                recall = data["Correctness"]["Overall Correctness"]["Overall Recall"]
                f1_score = data["Correctness"]["Overall Correctness"]["Overall F1 Score"]
                threshold = extract_threshold(filename)
                
                if threshold is not None:
                    thresholds.append(threshold)
                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1_score)
                    data_files.append(data)


    # Sort values by threshold
    sorted_indices = sorted(range(len(thresholds)), key=lambda k: thresholds[k])
    sorted_thresholds = [thresholds[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    sorted_precisions = [precisions[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    sorted_data_files = [data_files[i] for i in sorted_indices]


    # Plot threshold vs values
    y_min = min(sorted_accuracies + sorted_precisions + sorted_recalls + sorted_f1_scores)
    y_max = max(sorted_accuracies + sorted_precisions + sorted_recalls + sorted_f1_scores)
    y_ticks = [y * 0.01 for y in range(int(y_min // 0.01), int(y_max // 0.01) + 2)]
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_thresholds, sorted_accuracies, linestyle='-', marker='.', markersize=5, color='blue', label='Overall Accuracy')
    plt.plot(sorted_thresholds, sorted_precisions, linestyle='-', marker='.', markersize=5, color='green', label='Overall Precision')
    plt.plot(sorted_thresholds, sorted_recalls, linestyle='-', marker='.', markersize=5, color='red', label='Overall Recall')
    plt.plot(sorted_thresholds, sorted_f1_scores, linestyle='-', marker='.', markersize=5, color='purple', label='Overall F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Metrics')
    plt.title(f'Equation Similarity ({alg_num_threshold}, {alg_direction}): Metrics vs Threshold')
    plt.xticks(range(int(min(thresholds)), int(max(thresholds))+1, 5))
    plt.yticks(y_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plot_filename = os.path.join(PLOT_PATH, f'equation_similarity_{alg_num_threshold}_{alg_direction}_metrics_plot.png')
    plt.savefig(plot_filename)
    plt.show()


    # Find the threshold with the maximum F1 score
    max_f1_index = sorted_f1_scores.index(max(sorted_f1_scores))
    max_f1_threshold = sorted_thresholds[max_f1_index]
    max_f1_data = sorted_data_files[max_f1_index]


    # Extract box plot data from the JSON file with the maximum F1 score threshold
    aggregate_stats = max_f1_data["Correctness"]["Aggregate Correctness Statistics"]
    box_plot_data = [
        {
            'label': "Accuracy",
            'whislo': aggregate_stats["Accuracy"]["Lowest"],
            'q1': aggregate_stats["Accuracy"]["25th Quartile (Q1)"],
            'med': aggregate_stats["Accuracy"]["Median"],
            'q3': aggregate_stats["Accuracy"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["Accuracy"]["Highest"],
            'fliers': []
        },
        {
            'label': "Precision",
            'whislo': aggregate_stats["Precision"]["Lowest"],
            'q1': aggregate_stats["Precision"]["25th Quartile (Q1)"],
            'med': aggregate_stats["Precision"]["Median"],
            'q3': aggregate_stats["Precision"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["Precision"]["Highest"],
            'fliers': []
        },
        {
            'label': "Recall",
            'whislo': aggregate_stats["Recall"]["Lowest"],
            'q1': aggregate_stats["Recall"]["25th Quartile (Q1)"],
            'med': aggregate_stats["Recall"]["Median"],
            'q3': aggregate_stats["Recall"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["Recall"]["Highest"],
            'fliers': []
        },
        {
            'label': "F1 Score",
            'whislo': aggregate_stats["F1 Score"]["Lowest"],
            'q1': aggregate_stats["F1 Score"]["25th Quartile (Q1)"],
            'med': aggregate_stats["F1 Score"]["Median"],
            'q3': aggregate_stats["F1 Score"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["F1 Score"]["Highest"],
            'fliers': []
        }
    ]


    # Plot box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bxp(box_plot_data)
    ax.set_title(f'Aggregate Correctness Statistics Box Plot at Threshold {max_f1_threshold}')
    ax.set_ylabel('Values')
    ax.grid(True, linestyle='--', linewidth=0.5)
    box_plot_filename = os.path.join(PLOT_PATH, f'equation_similarity_{alg_num_threshold}_{alg_direction}_box_plot.png')
    plt.savefig(box_plot_filename)
    plt.show()


    # Prepare data for number line plots
    accuracy_data = []
    precision_data = []
    recall_data = []
    f1_score_data = []
    article_ids = []
    articles = max_f1_data["Results"]

    for article_id, stats in articles.items():
        article_ids.append(article_id)
        accuracy_data.append(stats["Accuracy"])
        precision_data.append(stats["Precision"])
        recall_data.append(stats["Recall"])
        f1_score_data.append(stats["F1 Score"])


    # Plot line plots
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(accuracy_data, [1]*len(accuracy_data), 'x', label='Accuracy', color='blue')
    ax.plot(precision_data, [2]*len(precision_data), 'x', label='Precision', color='green')
    ax.plot(recall_data, [3]*len(recall_data), 'x', label='Recall', color='red')
    ax.plot(f1_score_data, [4]*len(f1_score_data), 'x', label='F1 Score', color='purple')
    ax.set_xlabel('Values')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.set_title(f'Aggregate Correctness Statistics Line Plot at Threshold {max_f1_threshold}')
    line_plot_filename = os.path.join(PLOT_PATH, f'equation_similarity_{alg_num_threshold}_{alg_direction}_line_plot.png')
    plt.savefig(line_plot_filename)
    plt.show()
    plt.show()



# Plot naive bayes results
def plot_naive_bayes():

    # Extract threshold 
    def extract_threshold(filename):
        # Extract the numeric value from the filename
        match = re.search(fr'naive_bayes_(\d+\.?\d*)_', filename)
        if match:
            return float(match.group(1))
        return None


    # Extract data from json files
    threshold_data = defaultdict(lambda: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []})


    for filename in os.listdir(NAIVE_BAYES_PATH):
        if filename.endswith('.json'):
            file_path = os.path.join(NAIVE_BAYES_PATH, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                accuracy = data["Correctness"]["Overall Correctness"]["Overall Accuracy"]
                precision = data["Correctness"]["Overall Correctness"]["Overall Precision"]
                recall = data["Correctness"]["Overall Correctness"]["Overall Recall"]
                f1_score = data["Correctness"]["Overall Correctness"]["Overall F1 Score"]
                threshold = extract_threshold(filename)
                
                if threshold is not None:
                    threshold_data[threshold]['accuracy'].append(accuracy)
                    threshold_data[threshold]['precision'].append(precision)
                    threshold_data[threshold]['recall'].append(recall)
                    threshold_data[threshold]['f1_score'].append(f1_score)


    # Calculate average values for each threshold
    averaged_data = []
    for threshold, values in threshold_data.items():
        avg_accuracy = sum(values['accuracy']) / len(values['accuracy'])
        avg_precision = sum(values['precision']) / len(values['precision'])
        avg_recall = sum(values['recall']) / len(values['recall'])
        avg_f1_score = sum(values['f1_score']) / len(values['f1_score'])
        
        averaged_data.append({
            'threshold': threshold,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1_score
        })

    # Sort by threshold
    sorted_averaged_data = sorted(averaged_data, key=lambda x: x['threshold'])

    # Extract sorted values
    sorted_thresholds = [d['threshold'] for d in sorted_averaged_data]
    sorted_accuracies = [d['accuracy'] for d in sorted_averaged_data]
    sorted_precisions = [d['precision'] for d in sorted_averaged_data]
    sorted_recalls = [d['recall'] for d in sorted_averaged_data]
    sorted_f1_scores = [d['f1_score'] for d in sorted_averaged_data]


    # Plot threshold vs values
    y_min = min(sorted_accuracies + sorted_precisions + sorted_recalls + sorted_f1_scores)
    y_max = max(sorted_accuracies + sorted_precisions + sorted_recalls + sorted_f1_scores)
    y_ticks = [y * 0.01 for y in range(int(y_min // 0.01), int(y_max // 0.01) + 2)]
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_thresholds, sorted_accuracies, linestyle='-', marker='.', markersize=5, color='blue', label='Overall Accuracy')
    plt.plot(sorted_thresholds, sorted_precisions, linestyle='-', marker='.', markersize=5, color='green', label='Overall Precision')
    plt.plot(sorted_thresholds, sorted_recalls, linestyle='-', marker='.', markersize=5, color='red', label='Overall Recall')
    plt.plot(sorted_thresholds, sorted_f1_scores, linestyle='-', marker='.', markersize=5, color='purple', label='Overall F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Metrics')
    plt.title(f'Naive Bayes: Metrics vs Training Set Percentage')
    plt.xticks(range(int(0), int(100)+1, 5))
    plt.yticks(y_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plot_filename = os.path.join(PLOT_PATH, f'naive_bayes_metrics_plot.png')
    plt.savefig(plot_filename)
    plt.show()


    # Find the threshold with the maximum F1 score
    max_f1_index = sorted_f1_scores.index(max(sorted_f1_scores))
    max_f1_percentage = sorted_thresholds[max_f1_index]
    max_f1_data = None
    for filename in os.listdir(NAIVE_BAYES_PATH):
        if filename.endswith('.json') and f'naive_bayes_{int(max_f1_percentage)}_' in filename:
            file_path = os.path.join(NAIVE_BAYES_PATH, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                max_f1_data = data
            

    # Extract box plot data from the JSON file with the maximum F1 score threshold
    aggregate_stats = max_f1_data["Correctness"]["Aggregate Correctness Statistics"]
    box_plot_data = [
        {
            'label': "Accuracy",
            'whislo': aggregate_stats["Accuracy"]["Lowest"],
            'q1': aggregate_stats["Accuracy"]["25th Quartile (Q1)"],
            'med': aggregate_stats["Accuracy"]["Median"],
            'q3': aggregate_stats["Accuracy"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["Accuracy"]["Highest"],
            'fliers': []
        },
        {
            'label': "Precision",
            'whislo': aggregate_stats["Precision"]["Lowest"],
            'q1': aggregate_stats["Precision"]["25th Quartile (Q1)"],
            'med': aggregate_stats["Precision"]["Median"],
            'q3': aggregate_stats["Precision"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["Precision"]["Highest"],
            'fliers': []
        },
        {
            'label': "Recall",
            'whislo': aggregate_stats["Recall"]["Lowest"],
            'q1': aggregate_stats["Recall"]["25th Quartile (Q1)"],
            'med': aggregate_stats["Recall"]["Median"],
            'q3': aggregate_stats["Recall"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["Recall"]["Highest"],
            'fliers': []
        },
        {
            'label': "F1 Score",
            'whislo': aggregate_stats["F1 Score"]["Lowest"],
            'q1': aggregate_stats["F1 Score"]["25th Quartile (Q1)"],
            'med': aggregate_stats["F1 Score"]["Median"],
            'q3': aggregate_stats["F1 Score"]["75th Quartile (Q3)"],
            'whishi': aggregate_stats["F1 Score"]["Highest"],
            'fliers': []
        }
    ]


    # Plot box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bxp(box_plot_data)
    ax.set_title(f'Aggregate Correctness Statistics Box Plot at {max_f1_percentage}% Training ({max_f1_data["Correctness"]["Number of articles used"]} articles tested)')
    ax.set_ylabel('Values')
    ax.grid(True, linestyle='--', linewidth=0.5)
    box_plot_filename = os.path.join(PLOT_PATH, f'naive_bayes_box_plot.png')
    plt.savefig(box_plot_filename)
    plt.show()


    # Prepare data for number line plots
    accuracy_data = []
    precision_data = []
    recall_data = []
    f1_score_data = []
    article_ids = []
    articles = max_f1_data["Results"]

    for article_id, stats in articles.items():
        article_ids.append(article_id)
        accuracy_data.append(stats["Accuracy"])
        precision_data.append(stats["Precision"])
        recall_data.append(stats["Recall"])
        f1_score_data.append(stats["F1 Score"])


    # Plot line plots
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(accuracy_data, [1]*len(accuracy_data), 'x', label='Accuracy', color='blue')
    ax.plot(precision_data, [2]*len(precision_data), 'x', label='Precision', color='green')
    ax.plot(recall_data, [3]*len(recall_data), 'x', label='Recall', color='red')
    ax.plot(f1_score_data, [4]*len(f1_score_data), 'x', label='F1 Score', color='purple')
    ax.set_xlabel('Values')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.set_title(f'Aggregate Correctness Statistics Line Plot at {max_f1_percentage}% Training ({max_f1_data["Correctness"]["Number of articles used"]} articles tested)')
    line_plot_filename = os.path.join(PLOT_PATH, f'naive_bayes_line_plot.png')
    plt.savefig(line_plot_filename)
    plt.show()
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Which results to plot")
    parser.add_argument("-r", "--results", required=True,
    choices=['equation_similarity_1_greater', 'equation_similarity_1_lesser',
             'equation_similarity_2_greater', 'equation_similarity_2_lesser',
             'naive_bayes'],
    help="Which results to plot : ['equation_similarity_1_greater', 'equation_similarity_1_lesser', 'equation_similarity_2_greater', 'equation_similarity_2_lesser', 'naive_bayes']")
    args = parser.parse_args()

    argument = args.results
    if argument == 'equation_similarity_1_greater':
        # equation_similarity_1_greater()
        plot_equation_similarity(1, 'greater')
    elif argument == 'equation_similarity_1_lesser':
        plot_equation_similarity(1, 'lesser')
    elif argument == 'equation_similarity_2_greater':
        plot_equation_similarity(2, 'greater')
    elif argument == 'equation_similarity_2_lesser':
        plot_equation_similarity(2, 'lesser')
    elif argument == 'naive_bayes':
        plot_naive_bayes()