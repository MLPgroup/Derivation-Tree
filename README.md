# Summarizing and Extracting Derivation Trees from Mathematical Texts

## Instructions to Run Algorithm
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
- - - - 
## System Requirements
Python 3
- - - -

# Finding the most important equation

## Instructions to Run Algorithm
```
python3 derivation_tree.py
```
- - - -
## File System
### derivation_tree.py
- entry point that runs all code written in the following files
### articles.json
- json file to hold information about each article
### article_parser.py
- util file that parses the json file and returns a dictionary of the articles
### important_equation.py
- util file that runs the important equation custom algorithm on an article
- - - - 
## System Requirements
Python 3.5 (can run on earlier versions but is untested and requires a few changes)
- - - -
