# Import modules
import article_parser
import important_equation


"""
run_derivation_algo()
Input: none
Return: none
Function: Find and print the most important equation for the articles listed in the articles.json file
"""
def run_derivation_algo():
    articles = article_parser.get_manually_parsed_articles()
    for article_id, article in articles.items():
        most_important = important_equation.get_most_important_equation(article)
        print(f"Article ID: {article_id} => most important equation(s): {most_important}")


"""
Entry point for derivation_tree.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    run_derivation_algo()
