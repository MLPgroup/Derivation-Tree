import json
import os

# ------------------------------- # False Pos / False Neg --------------------------------
# Description: Compares Adjacency List (Manually Parsed Data) with Bard's output to output false pos/neg
# @Param    json_data = Data from specified json file
#           target_article_id = ID of specified article
# ------------------------------------------------------------------------------------------

def compare_adjacency_lists(expected, actual):
    true_positives = {
        key: value
        for key, value in actual.items()
            if (key in expected and set(value) == set(expected[key]) and expected[key][0] != None and actual[key][0] != None)
    }

    true_negatives = {
        key: value
        for key, value in actual.items()
        if (key in expected and set(value) == set(expected[key]) and expected[key][0] == None and actual[key][0] == None)
    }

    false_positives = {
        key: value
        for key, value in actual.items()
        if key not in expected or (key in expected and set(value) != set(expected[key]))
    }

    false_negatives = {
        key: value
        for key, value in expected.items()
        if key not in actual or (key in actual and set(value) != set(actual[key]))
    }

    # Debugging
    # print("\nTN: ", true_negatives)
    # print("\nTP: ", true_positives)
    # print("\nFN: ", false_negatives)
    # print("\nFP: ", false_positives)

    return true_positives, true_negatives, false_positives, false_negatives


# ------------------------------- # Creates Adjacency List --------------------------------
# Description: Creates a Map of the Adjacency List data from json file
# @Param    json_data = Data from specified json file
#           target_article_id = ID of specified article
# ------------------------------------------------------------------------------------------

def find_adjacency_list(json_data, target_article_id):
    for article in json_data["Manually Parsed Articles"]:
        if article["Article ID"] == target_article_id:
            return article.get("Adjacency List", {})
    return {}

# -------------------------- # Accuracy, Precision, Recall for Bard ---------------------------
# Description: Identifies Accuracy, Precision, Recall for a specific article
# @Param    adjList = Adjacency List that was created from the tempGraphing file
# --------------------------------------------------------------------------------------------
def APR(adjList):
    # Get JSON file name from user input
    json_file_name = input("Enter the name of your JSON file (e.g., articles.json): ")

    # Construct the full path to the JSON file (assuming it's in the same directory as the script)
    script_directory = os.path.dirname(os.path.realpath(__file__))
    json_file_path = os.path.join(script_directory, json_file_name)

    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_name}' not found in the script's directory.")
        exit()

    # Load JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Get target Article ID from user input
    target_article_id = input("Enter the target Article ID: ")

    # Find the adjacency list based on the target Article ID
    result_article = find_adjacency_list(data, target_article_id)
    
    total_examples = len(result_article)

    # Compare Correct Adjacency List with Bard's data
    TP, TN, FP, FN= compare_adjacency_lists(result_article, adjList)
    
    # Calculating Accuracy, Precision, Recall
    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) != 0 else 0
    recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) != 0 else 0
    accuracy = (len(TP) + len(TN)) / total_examples

    print("\nPrecision: ", precision)
    print("\nAccuracy: ", accuracy)
    print("\nRecall: ", recall)

# --------------------------------------------------------------------------------------------

# Replace 'adjList' with the data from Bard
adjList = {
    "S0.E1": [None],
    "S0.E2": ["S0.E5"],
    "S0.E3": ["S0.E4"],
    "S0.E4": ["S0.E5", "S0.E7"],
    "S0.E5": [None],
    "S0.E6": ["S0.E7"],
    "S0.E7": ["S0.E8", "S0.E9", "S0.E10"],
    "S0.E8": [None],
    "S0.E9": [None],
    "S0.E10": [None]
}

modelAdjList = APR(adjList)  # Compares outputted adjacency list with manually parsed derivation links
