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

    # Iterate through all 'math' elements in the HTML
    for mathml in soup.find_all('math'):
        # Get equation ID and alt text attributes
        equation_id = mathml.get('id', '')
        alttext = mathml.get('alttext', '')

        # Check if the equation ID matches the defined pattern
        match = pattern.search(equation_id)
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

    return equations


"""
run_equation_similarity()
Input: none
Return: none
Function: Find the equations in articles and construct a tree depending on equation similarity
"""
def run_equation_similarity():
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    j = 0
    # Iterate through article IDs
    for i, cur_article_id in enumerate(article_ids):
        if i == 0:
            continue
        # Construct the HTML file path for the current article
        html_path = f'articles/{cur_article_id}.html'
    
        # Check if the HTML file exists
        if os.path.exists(html_path):
            # Read the content of the HTML file
            with open(f'articles/{cur_article_id}.html', 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # Extract equations from the HTML content
            equations = extract_equations(html_content)
            print(len(equations))
            j+=1    

            if j > 1:
                for equation_key, equation_data in equations.items():
                    print(f"Equation: {equation_key}")
                    for subequation in equation_data['equations']:
                        print(f"  Equation ID: {subequation['equation_id']}")
                    print("---")
                break
        else:
            print(f"HTML file {html_path} not found")

    return 0


"""
Entry point for equation_similarity.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    run_equation_similarity()