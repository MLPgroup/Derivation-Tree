import json
import preProcessing
import os

# ------------------------------- # False Pos / False Neg --------------------------------
# Description: Compares Adjacency List (Manually Parsed Data) with Algorithm output to output false pos/neg
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
# Description: Finds a Map of the Adjacency List data from json file
# @Param    json_data = Data from specified json file
#           target_article_id = ID of specified article
# ------------------------------------------------------------------------------------------

def find_adjacency_list(json_data, target_article_id):
    for article in json_data["Manually Parsed Articles"]:
        if article["Article ID"] == target_article_id:
            return article.get("Adjacency List", {})
    return {}

# ------------------------------- # Creates Equation ID List --------------------------------
# Description: Finds a Map of the eqID (mathML) List data from json file
# @Param    json_data = Data from specified json file
#           target_article_id = ID of specified article
# ------------------------------------------------------------------------------------------

def find_eqID_list(json_data, target_article_id):
    for article in json_data["Manually Parsed Articles"]:
        if article["Article ID"] == target_article_id:
            return article.get("Equation ID", [])
    return []

# ----------------------------- # Accuracy, Precision, Recall  ------------------------------
# Description: Identifies Accuracy, Precision, Recall for a specific article
# @Param        json_file_name: .json name
#               target_article_id: article ID
#               adjList = Adjacency List that was created from the tempGraphing file
# --------------------------------------------------------------------------------------------
def APR(json_file_name, target_article_id, adjList):
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

    # Find the adjacency list based on the target Article ID
    result_article = find_adjacency_list(data, target_article_id)
    
    total_examples = len(result_article)

    # Compare Correct Adjacency List 
    TP, TN, FP, FN= compare_adjacency_lists(result_article, adjList)
    
    # Calculating Accuracy, Precision, Recall
    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) != 0 else 0
    recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) != 0 else 0
    accuracy = (len(TP) + len(TN)) / total_examples

    # Debugging Precision, Accuracy, Recall
    #print("\nPrecision: ", precision)
    #print("\nAccuracy: ", accuracy)
    #print("\nRecall: ", recall)

    return precision, accuracy, recall

# --------------------------------------------------------------------------------------------
# Conveniently Check number of articles in articles.json vs any other database
# --------------------------------------------------------------------------------------------  

# Load the JSON data from the file
with open('articles.json', 'r') as file:    # Change to whichever json
    data = json.load(file)

# Get the list of manually parsed articles
articles = data.get('Manually Parsed Articles', [])     # Change to whichever section

# Get the count of items in the list
num_articles = len(articles)

# Print the count
print("Number of items in 'Manually Parsed Articles' section:", num_articles)

# --------------------------------------------------------------------------------------------
# (Gemini Testing)
# --------------------------------------------------------------------------------------------  

def GeminiTest():

    # Get JSON file name from user input
    json_file_name = input("Enter the name of your JSON file that holds Manually Parsed Articles (e.g., articles.json): ")

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

    # Find the eqeID list based on the target Article ID
    eqIDs = find_eqID_list(data, target_article_id)

    '''
    # Find the adjacency list based on the target Article ID
    adjacency_list = find_adjacency_list(data, target_article_id)
    '''

    print("Enter Gemini Data: ")
    adjList = {}
    while True:
        node = input("Enter Equation # (or 'done' to finish): ")
        if node.lower() == 'done':
            break
        neighbors = input("Enter Derivation Links (comma-separated, or 'none'): ").split(", ")
        if neighbors[0] == 'none':
            adjList[eqIDs[int(node)-1]] = [None]
        elif eqIDs[int(node)-1] in adjList:
            for n in neighbors:
                adjList[eqIDs[int(node)-1]].append(eqIDs[int(n)-1])
        else:           
            adjList[eqIDs[int(node)-1]] = [eqIDs[int(n)-1] for n in neighbors]

    precision, accuracy, recall = APR(json_file_name, target_article_id, adjList)  # Compares outputted adjacency list with manually parsed derivation links

    # Get JSON file name from user input
    existing_file = input("Enter the name of your JSON file to hold Gemini Data (e.g., GeminiOutput.json): ")

    # Who Tested 
    labeled_by = input("Enter Labeled By: ")

    graphs = {
            "Article ID": target_article_id,
            "Gemini's Adjacency List": adjList,
            "Accuracy": precision,
            "Precision": accuracy,
            "Recall": recall,
            "Tested by": labeled_by
        }



    # Check if the file exists
    if os.path.exists(existing_file):
        # Read existing JSON data from the file
        with open(existing_file, 'r') as file:
            data = json.load(file)

            # Check if "Gemini Data" key exists in the data. Changed from "Bard Data" (February 15th 2024)"
            if "Gemini Data" in data:
                data["Gemini Data"].append(graphs)  # Append the current graph to the existing data
            else:
                print("The existing JSON file does not contain 'Gemini Data'. Initializing it.")
                data["Gemini Data"] = [graphs]  # Initialize "Gemini Data" with the current graph

            # Write the updated JSON data to the specified file
            with open(existing_file, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"JSON data has been written to {existing_file}.")
    else:
        print(f"The specified file '{existing_file}' does not exist.")

# --------------------------------------------------------------------------------------------
# (Brute Force Testing)
# --------------------------------------------------------------------------------------------  

def brute_force_test():
     # Get JSON file name from user input
    json_file_name = 'articles.json'

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

    # Find the eqeID list based on the target Article ID
    eqIDs = find_eqID_list(data, target_article_id)

    '''
    # Find the adjacency list based on the target Article ID
    adjacency_list = find_adjacency_list(data, target_article_id)
    '''

    print("Enter Brute Force Data: ")
    adjList = {}
    while True:
        node = input("Enter Equation # (or 'done' to finish): ")
        if node.lower() == 'done':
            break
        neighbors = input("Enter Derivation Links (comma-separated, or 'none'): ").split(", ")
        if neighbors[0] == 'none':
            adjList[eqIDs[int(node)-1]] = [None]
        elif eqIDs[int(node)-1] in adjList:
            for n in neighbors:
                adjList[eqIDs[int(node)-1]].append(eqIDs[int(n)-1])
        else:           
            adjList[eqIDs[int(node)-1]] = [eqIDs[int(n)-1] for n in neighbors]

    precision, accuracy, recall = APR(json_file_name, target_article_id, adjList)  # Compares outputted adjacency list with manually parsed derivation links

    # Get JSON file name from user input
    existing_file = 'BruteForceOutput.json'

    # Who Tested 
    labeled_by = 'Brian Kim'

    graphs = {
            "Article ID": target_article_id,
            "Brute Force's Adjacency List": adjList,
            "Accuracy": precision,
            "Precision": accuracy,
            "Recall": recall,
            "Tested by": labeled_by
        }

    # Read existing JSON data from the file
    with open(existing_file, 'r') as file:
        data = json.load(file)

        # Check if "Gemini Data" key exists in the data. Changed from "Bard Data" (February 15th 2024)"
        if "Brute Force Data" in data:
            data["Brute Force Data"].append(graphs)  # Append the current graph to the existing data
        else:
            print("The existing JSON file does not contain 'Brute Force Data'. Initializing it.")
            data["Brute Force Data"] = [graphs]  # Initialize "Gemini Data" with the current graph

        # Write the updated JSON data to the specified file
        with open(existing_file, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"JSON data has been written to {existing_file}.")


# --------------------------------------------------------------------------------------------
# Main Function 
# --------------------------------------------------------------------------------------------  
brute_force_test()
# GeminiTest()