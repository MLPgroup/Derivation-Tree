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
    algo_most_important_equations = []
    labeled_most_important_equations = []
    for article_id, article in articles.items():
        cur_most_important_equation = important_equation.get_most_important_equation(article)
        print(f"Article ID: {article_id} => most important equation(s): {cur_most_important_equation}")
        algo_most_important_equations.append(cur_most_important_equation)
        labeled_most_important_equations.append(article['Most Important Equation'])
    algo_conf_matrix, algo_accuracy, algo_precision, algo_recall, algo_specificity, algo_f1_score = important_equation.get_algo_correctness(labeled_most_important_equations, algo_most_important_equations)
    print("*-----------------------------------------------------------*")
    print("Important Equation Algorithm Correctness: ")
    print("Accuracy:", algo_accuracy)
    print("Precision:", algo_precision)
    print("Recall:", algo_recall)
    print("Specificity:", algo_specificity)
    print("F1 Score:", algo_f1_score)

"""
Entry point for derivation_tree_algo.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    run_derivation_algo()
