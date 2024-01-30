import json
import preProcessing
import os  # Import the os module to check if the file exists

# --------------------------------- # Article Additions  ----------------------------------
# Description: Creates json data to add properly formatted article data to specified json file
# @Param    none
# ------------------------------------------------------------------------------------------
def create_graph():
    url = 'file:///C:/Users/brian/Desktop/MLP/Derivation-Tree/articles/0907.3505.html'      # Original Mathematical Document
    mathML = preProcessing.eqExtract(url)                                                   # Extract Block Equations
    eqIDs = preProcessing.idExtract(mathML)                                                 # Extract all Block Equation IDs
    print(eqIDs)                                               
    article_id = input("Enter Article ID: ")
    
    # Input adjacency list as a dictionary
    adjacency_list = {}
    while True:
        node = input("Enter Equation # (or 'done' to finish): ")
        if node.lower() == 'done':
            break
        neighbors = input("Enter Derivation Links (comma-separated, or 'none'): ").split(", ")
        if neighbors[0] == 'none':
            adjacency_list[eqIDs[int(node)-1]] = [None]
        elif eqIDs[int(node)-1] in adjacency_list:
            for n in neighbors:
                adjacency_list[eqIDs[int(node)-1]].append(eqIDs[int(n)-1])
        else:           
            adjacency_list[eqIDs[int(node)-1]] = [eqIDs[int(n)-1] for n in neighbors]

    Equation_Number = {}

    for index, eq in enumerate(eqIDs):
        Equation_Number[eq] = index + 1

    Most_Important_Equation_Index = input("Enter Most Important Equation: ")
    Most_Important_Equation = eqIDs[int(Most_Important_Equation_Index)-1]

    labeled_by = input("Enter Labeled By: ")

    graph = {
        "Article ID": article_id,
        "Equation ID": eqIDs,
        "Adjacency List": adjacency_list,
        "Equation Number": Equation_Number,
        "Most Important Equation": Most_Important_Equation,
        "Labeled by": labeled_by
    }
    return graph

# --------------------------------------------------------------------------------------------
# Main

# Ask the user for the existing JSON file
existing_file = input("Enter the existing JSON file name (e.g., existing_data.json): ")

# List to store multiple graphs
graphs = []

# Get the number of new graphs from the user
num_graphs = int(input("Enter the number of new articles: "))

# Check if the file exists
if os.path.exists(existing_file):
    # Read existing JSON data from the file
    with open(existing_file, 'r') as file:
        data = json.load(file)

    # Check if "Manually Parsed Articles" key exists in the data
    if "Manually Parsed Articles" in data:
        # Append new graphs to the existing data within "Manually Parsed Articles"
        graphs.extend(create_graph() for _ in range(num_graphs))
        data["Manually Parsed Articles"].extend(graphs)
    else:
        print("The existing JSON file does not contain 'Manually Parsed Articles'. Initializing it.")

    # Write the updated JSON data to the specified file
    with open(existing_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"JSON data has been written to {existing_file}.")
else:
    print(f"The specified file '{existing_file}' does not exist.")