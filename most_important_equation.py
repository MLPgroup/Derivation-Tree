'''
Description: Python code to parse json file and store manually parsed articles
Author: Vishesh Prasad
Modification Log:
    October 22, 2023: created file and wrote initial algorithm
'''



"""
Function: get_most_important_equation(article)
Input: article -- dictionary with following values:
                - Article ID: (int)
                - Equation ID: (list)
                - Adjacency List (dict)
"""
def get_most_important_equation(article):
    # Get required information
    equation_list = set(article['Equation ID'])
    adjacency_list =article['Adjacency List']
