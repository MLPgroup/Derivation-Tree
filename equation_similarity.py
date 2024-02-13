'''
Description: Python code to parse article html and extract equations
Author: Vishesh Prasad
Modification Log:
    February 10, 2024: create file and extract equations from html successfully 
'''

from bs4 import BeautifulSoup
import os
import re
import article_parser


"""
extract_equations(html_content)
Input: html_content -- html content for current article that needs to be parsed
Return: equations -- equations that were found in the article
Function: Find and return all the equations, their ids, and equation content from the given article
"""
def extract_equations(html_content):
    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Dictionary to store equations
    equations = {}

    # Define the pattern to match equations
    pattern = re.compile(r'S(\d+)\.E(\d+)')
    # pattern_2 = re.compile(r'S(\d+)\.Ex(\d+)')

    # Iterate through all 'math' elements in the HTML
    for mathml in soup.find_all('math'):
        # Get equation ID and alt text attributes
        equation_id = mathml.get('id', '')
        alttext = mathml.get('alttext', '')

        # Check if the equation ID matches the defined pattern
        match = pattern.search(equation_id)
        # match_2 = pattern_2.search(equation_id)
        if match:
            # Extract section and equation numbers from the matched pattern
            section_number, equation_number = match.groups()
            equation_key = f"S{section_number}.E{equation_number}"

             # Create an entry in the dictionary for the equation if not present
            if equation_key not in equations:
                equations[equation_key] = {
                    'section_number': int(section_number),
                    'equation_number': int(equation_number),
                    'equations': [],
                }

            # Add the equation details to the list of equations for the current key
            equations[equation_key]['equations'].append({
                'mathml': str(mathml),
                'equation_id': equation_id,
                'alttext': alttext,
            })
        # elif len(equations) == 0 and match_2:
        #     # Extract section and equation numbers from the matched pattern
        #     section_number, equation_number = match_2.groups()
        #     equation_key = f"S{section_number}.E{equation_number}"

        #     # Create an entry in the dictionary for the equation if not present
        #     if equation_key not in equations:
        #         equations[equation_key] = {
        #             'section_number': int(section_number),
        #             'equation_number': int(equation_number),
        #             'equations': [],
        #         }

        #     # Add the equation details to the list of equations for the current key
        #     equations[equation_key]['equations'].append({
        #         'mathml': str(mathml),
        #         'equation_id': equation_id,
        #         'alttext': alttext,
        #     })

    return equations


"""
combine_sub_equations(equation)
Input: equation -- one equation in the article and all of its sub equations
Return: combined_mathml -- string with combined mathml for one equation
Function: Combine mathml for equation and all sub equations to compare with other equations
"""
def combine_sub_equations(equation):
    # Combine MathMLs of all sub-equations
    combined_mathml = ''.join(sub_equation['mathml'] for sub_equation in equation['equations'])
    return combined_mathml


"""
compute_symbol_percentage(equation1, equation2)
Input: equation1 -- mathml for one equation
       equation2 -- mathml for another equation
Return: percentage_equation1_in_equation2, percentage_equation2_in_equation1 - equation similarity percentages
Function: Compute the percentages of symbols in equation1 that are found in equation2 and vice verse
"""
def compute_symbol_percentage(equation1, equation2):
    set_equation1 = set(equation1)
    set_equation2 = set(equation2)

    percentage_equation1_in_equation2 = (len(set_equation1.intersection(set_equation2)) / len(set_equation1)) * 100
    percentage_equation2_in_equation1 = (len(set_equation2.intersection(set_equation1)) / len(set_equation2)) * 100

    return percentage_equation1_in_equation2, percentage_equation2_in_equation1


"""
equation_similarity_percentages(equations)
Input: equations -- equations found in article
Return: similarity_matrix -- [i][j] = percentage of equation i that is found in equation j
Function: Find similarity percentages between all equations
"""
def equation_similarity_percentages(equations):
    # Set up similarity matrix
    num_equations = len(equations)
    similarity_matrix = [[0.0] * num_equations for _ in range(num_equations)]

    # Combine mathml
    combined_mathml = [combine_sub_equations(equations[cur_equation]) for cur_equation in equations]

    # Compute similarity percentages
    for i in range(num_equations - 1):
        equation_i = combined_mathml[i]
        for j in range(i + 1, num_equations):
            equation_j = combined_mathml[j]

            # Compute percentage similar
            percentage_i_in_j, percentage_j_in_i = compute_symbol_percentage(equation_i, equation_j)

            # Store percentages in matrix
            similarity_matrix[i][j] = percentage_i_in_j
            similarity_matrix[j][i] = percentage_j_in_i

    return similarity_matrix


"""
run_equation_similarity()
Input: none
Return: none
Function: Find the equations in articles and construct a tree depending on equation similarity
"""
def run_equation_similarity():
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    # extracted_equations = []

    # Iterate through article IDs
    for i, (cur_article_id, cur_article) in enumerate(article_ids.items()):
        # Construct the HTML file path for the current article
        html_path = f'articles/{cur_article_id}.html'
    
        # Check if the HTML file exists
        if os.path.exists(html_path):
            # Read the content of the HTML file
            with open(f'articles/{cur_article_id}.html', 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # Extract equations from the HTML content
            equations = extract_equations(html_content)

            # If extracted correctly, compute similarity
            if len(cur_article["Equation ID"]) == len(equations) and all(cur_equation in cur_article["Equation ID"] for cur_equation in equations):
                computed_similarity = equation_similarity_percentages(equations)
                print(cur_article_id)
                for row in computed_similarity:
                    print(' '.join(f'{percentage:.2f}' for percentage in row))
                return 0


            
            """ Debugging: """
            # for equation_key, equation_data in equations.items():
            #     print(f"Equation: {equation_key}")
            #     for subequation in equation_data['equations']:
            #         print(f"  Equation ID: {subequation['equation_id']}")
            #     print("---")

            # if len(cur_article["Equation ID"]) != len(equations):
            #     print(f"Equation ID: {cur_article_id} not same")
            #     print(len(cur_article["Equation ID"]))
            #     print(len(equations))
            #     print(cur_article["Equation ID"])
            #     for equation_key, equation_data in equations.items():
            #         print(f"Equation: {equation_key}")
            #         for subequation in equation_data['equations']:
            #             print(f"  Equation ID: {subequation['equation_id']}")
            #         print("---")

            # else:
            #     continue
            #     print(f"Equation ID: {cur_article_id} same")

            break

        else:
            # No html for article found
            continue
            print(f"HTML file {html_path} not found")

    return 0


"""
Entry point for equation_similarity.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    run_equation_similarity()