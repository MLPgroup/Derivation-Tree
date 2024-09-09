# Summarizing and Extracting Derivation Graphs from Mathematical Texts
- - - -
# Building the Derivation Graph Using Token Similarity, Naive Bayes, Brute Force, or LLMs

## Instructions to Run Algorithms
To run the models:
```
python3 derivation_graph.py -a [bayes, token, brute, gemini]
```
or
```
python3 derivation_graph.py --algorithm [bayes, token, brute, gemini]
```
'bayes' - will run the Naive Bayes model \
'token' - will run the string similarity analytical model \
'brute' - will run the brute force model \
'gemini' - will run the Gemini LLM model
<br/>
## Instructions to Plot Results
To plot how results shift with certain hyper-parameters for some models:
```
python3 plot_results.py -r [token_similarity_1_greater, token_similarity_1_lesser, token_similarity_2_greater, token_similarity_2_lesser, naive_bayes] 
```
or 
```
python3 plot_results.py --results [token_similarity_1_greater, token_similarity_1_lesser, token_similarity_2_greater, token_similarity_2_lesser, naive_bayes]
```
- - - -
## Output
Running the above command will output the results of the model specified, which include the predicted adjacency list, along with the correctness metrics (for individual articles as well as the aggregate). The results are in a formatted JSON file under /outputs. The specific output path will be printed to the terminal.
- - - -
## File System
### derivation_graph.py
- Entry point that runs all the code written in the file. It handles most of the pre-processing for the models and calls the specific code for said model. It also handles computing the correctness of each metric and handing off the output writing.
### token_similarity.py
- Util file that holds functions specific to running the token similarity model
### naive_bayes.py
- Util file that holds functions specific to running the naive bayes model
### gemini.py
- Util file that holds functions specific to running the Gemini LLM model
### brute_force.py
- Util file that holds functions specific to running the brute force model
### articles.json
- JSON file to hold information about each article
### article_parser.py
- Util file that parses the json file and returns a dictionary of the articles
### results_output.py
- Util file that writes output of the models to the correct location and with correct formatting
### plot_results.py
- Util file that aggregates the outputs of certain algorithms and places them in the outputs folder
### /articles (folder)
- Holds the html for the articles that are in the articles.json file
### /outputs (folder)
- Holds the output for the programs being run
- - - - 
## System Requirements
Python 3.10. See requirements.txt \
For running the Gemini model, first install the Gemini API SDK
```
pip install -q -U google-generativeai
```
You then have to create a Google AI studio API key and export it to your environment
```
export API_KEY=<YOUR_API_KEY>
```
For the latest installation steps and requirements, visit the Google Gemini documentation [https://ai.google.dev/gemini-api/docs/quickstart?lang=python](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)
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
Running the above command will output the results of the algorithm specified, which include the predicted most important equation, along with the correctness metrics. The results are in a formatted JSON file under /outputs. The specific output path will be printed to the terminal.
- - - -
## File System
### articles.json
- JSON file to hold information about each article
### article_parser.py
- Util file that parses the json file and returns a dictionary of the articles
### important_equation.py
- Util file that runs the important equation custom algorithm on an article and outputs results to the terminal
- Runs the correctness script to test the algorithm's output
### results_output.py
- Util file that writes output of the algorithm to the correct location and with correct formatting
### /outputs (folder)
- Holds the output for the programs being run
- - - - 
## System Requirements
Python 3.10. See requirements.txt.
- - - -
&nbsp;
- - - -
# Instructions to Run Sub-Tree Similarity Algorithm
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