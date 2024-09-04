'''
Description: Python utils file to use token similarity to get derivation graphs
Author: Vishesh Prasad
Modification Log:
    August 18, 2024: create file and transfer code in
'''



# Import Modules



"""
combine_sub_equations(equation)
Input: equation -- one equation in the article and all of its sub equations
Return: combined_mathml -- string with combined mathml for one equation
Function: Combine mathml for equation and all sub equations to compare with other equations
"""
def combine_sub_equations(equation):
    # Combine MathMLs of all sub-equations
    combined_mathml = ''.join(sub_equation['alttext'] for sub_equation in equation['equations'])
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
token_similarity_percentages(equations)
Input: equations -- equations found in article
Return: similarity_matrix -- [i][j] = percentage of equation i that is found in equation j
        equation_order -- order of equations in matrix
Function: Find similarity percentages between all equations
"""
def token_similarity_percentages(equations):
    # Set up similarity matrix
    num_equations = len(equations)
    similarity_matrix = [[0.0] * num_equations for _ in range(num_equations)]

    # Combine mathml
    combined_mathml = [combine_sub_equations(equations[cur_equation]) for cur_equation in equations]
    equation_order = [cur_equation for cur_equation in equations]

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

    return similarity_matrix, equation_order



"""
token_similarity_adjacency_list(similarity_matrix, equation_order, similarity_threshold)
Input: similarity_matrix -- [i][j] = percentage of equation i that is found in equation j
        equation_order -- order of equations in matrix
        similarity_threshold -- threshold of matrix to determine if two equations are similar or not
        similarity_direction -- direction of similarity check to add edge
        similarity_strictness -- integer value (x = 0, 1, 2) to force minimum x number of similarity values to be greater than the threshold in edge determination
Return: equation_adjacency_list -- adjacency list computed using 
Function: Construct an adjacency list from the similarity matrix
"""
def token_similarity_adjacency_list(similarity_matrix, equation_order, similarity_threshold, similarity_direction, similarity_strictness):
    num_equations = len(equation_order)
    equation_adjacency_list = {equation_order[i]: [] for i in range(num_equations)}

    # Iterate through similarity matrix
    for i in range(num_equations - 2, -1, -1):
        for j in range(num_equations - 1, i - 1, -1):
            match similarity_strictness:
                # Strictness
                # Case 0 = no restriction
                # Case 1 = at least one cell has to be greater than the threshold
                # Case 2 = both cells have to be greater than the threshold
                case 0:
                    if similarity_direction == 'greater':
                        if similarity_matrix[i][j] > similarity_matrix[j][i]:
                            equation_adjacency_list[equation_order[i]].append(equation_order[j])
                        else:
                            equation_adjacency_list[equation_order[j]].append(equation_order[i])
                    else:
                        if similarity_matrix[i][j] < similarity_matrix[j][i]:
                            equation_adjacency_list[equation_order[i]].append(equation_order[j])
                        else:
                            equation_adjacency_list[equation_order[j]].append(equation_order[i])
                
                case 1: 
                    if similarity_matrix[i][j] >= similarity_threshold or similarity_matrix[j][i] >= similarity_threshold:
                        if similarity_direction == 'greater':
                            if similarity_matrix[i][j] > similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])
                        else:
                            if similarity_matrix[i][j] < similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])
                case 2:
                    if similarity_matrix[i][j] >= similarity_threshold and similarity_matrix[j][i] >= similarity_threshold:
                        if similarity_direction == 'greater':
                            if similarity_matrix[i][j] > similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])
                        else:
                            if similarity_matrix[i][j] < similarity_matrix[j][i]:
                                equation_adjacency_list[equation_order[i]].append(equation_order[j])
                            else:
                                equation_adjacency_list[equation_order[j]].append(equation_order[i])

    # Formatting
    for i in range(num_equations):
        if len(equation_adjacency_list[equation_order[i]]) == 0:
            equation_adjacency_list[equation_order[i]] = [None]

    # Return adjacency list
    return equation_adjacency_list