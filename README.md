# Summarizing and Extracting Derivation Trees from Mathematical Texts

## Instructions to Run Sub-Tree Similarity Algorithm
```
python preProcessing.py
```
- - - -
## File System
### preProcessing.py
- Processes html data into a useable format for tempGraphing.py
### tempGraphing.py
- Edge Mapping Logic for mathML components
### mathMLtoOP.py
- Helper file for converting mathML components to OP Tree
- File Credits: Sumedh Vemuganti
### articleScript.py
- Automates general article info (Article ID, Nodes, Edges) to proper json format
### bardTest.py
- Automates comparing Bard Derivation Tree output with expected Adjacency List
- - - - 
## System Requirements
Python 3
- - - -
&nbsp;
- - - -
# Finding the most important equation

## Instructions to Run Algorithm
```
python3 important_equation.py -a [bfs, dfs]
```
or
```
python3 important_equation.py --algorithm [bfs, dfs]
```
- - - -
## Output
Running the above command will output 2 sections into the terminal. The first section holds, for each article in the JSON file below, the article id, the most important equation found by the algorithm for that article, and the labeled most important equation. The second section holds the output of the correctness script with metrics used to test the correctness of the algorithm.
- - - -
## File System
### articles.json
- JSON file to hold information about each article
### article_parser.py
- Util file that parses the json file and returns a dictionary of the articles
### important_equation.py
- Util file that runs the important equation custom algorithm on an article and outputs results to the terminal
- Runs the correctness script to test the algorithm's output
### extra_articles.json
- JSON file to hold information about articles that don't have a corresponding html file on the corpus
- - - - 
## System Requirements
Python 3.10. See requirements.txt.
- - - -
&nbsp;
- - - -
# Building the Derivation Tree Using Token Similarity and Naive Bayes

## Instructions to Run Algorithm
```
python3 derivation_graph.py -a [bayes, token]
```
or
```
python3 derivation_graph.py --algorithm [bayes, token]
```
'bayes' - will run the Naive Bayes model
'token' - will run the string similarity analytical model
- - - -
## Output
Running the above command will output results of the correctness script with metrics used to test the correctness of the algorithm which was computed using the labeled adjacency list and the predicted adjacency list.
- - - -
## File System
### derivation_graph.py
- Entry point that runs all the code written in the file. It first extracts the equations from the html, then either computes the similarities between all equations in that were extracted or trains a Naive Bayes model, depending on the type specified by the command line argument. It finally constructs an adjacency list and compares it to the labeled adjacency list provided in articles.json.
### /articles (folder)
- Holds the html for the articles that are in the articles.json file
- - - - 
## System Requirements
Python 3.10. See requirements.txt
- - - -
