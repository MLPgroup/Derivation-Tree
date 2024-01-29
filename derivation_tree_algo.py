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
    article_id_correctness = []
    
    print("*-----------------------------------------------------------*")
    print("Important Equation Algorithm Output:")
    for article_id, article in articles.items():
        cur_most_important_equation = important_equation.get_most_important_equation(article)
        print(f"Article ID: {article_id} => algo equation(s): {cur_most_important_equation} vs. labeled equation(s): {article['Most Important Equation']}")
        algo_most_important_equations.append(cur_most_important_equation)
        labeled_most_important_equations.append(article['Most Important Equation'])
        article_id_correctness.append(article_id)

    algo_conf_matrix, algo_accuracy, algo_precision, algo_recall, algo_specificity, algo_f1_score, algo_articles_used = important_equation.get_algo_correctness(labeled_most_important_equations, algo_most_important_equations, article_id_correctness)
    print("*-----------------------------------------------------------*")
    print("Important Equation Algorithm Correctness: ")
    print(f"Articles used for correctness calculations: {algo_articles_used}")
    print(f"Accuracy: {algo_accuracy:.8f}")
    print(f"Precision: {algo_precision:.8f}")
    print(f"Recall: {algo_recall:.8f}")
    print(f"Specificity: {algo_specificity:.8f}")
    print(f"F1 Score: {algo_f1_score:.8f}")
    print("*-----------------------------------------------------------*")

"""
Entry point for derivation_tree_algo.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    run_derivation_algo()
