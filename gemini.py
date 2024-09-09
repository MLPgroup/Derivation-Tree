import google.generativeai as genai
import os
import article_parser
import derivation_graph


# Model Configuration
# model = genai.GenerativeModel("text-embedding-004")



def parse_adjacency_list(text_response, equation_indexing):
    adjacency_list = {}
    
    # Split text_response into lines
    lines = text_response.strip().split('\n')

    # Iterate through each line
    for cur_line in lines:
        # Clean up input string
        cur_line = cur_line.rstrip(';').strip()
        # Format checking
        if '->' not in cur_line:
            return f"Error: Line '{cur_line}' is not correctly formatted (missing '->')."
        
        # Split input line
        part1, part2 = cur_line.split('->')

        # Clean up
        starting_node_index = part1.strip()
        # Format checking
        if not starting_node_index.isdigit():
            return f"Error: Invalid node '{starting_node_index}'. Nodes should be integers."
        
        
        starting_node = equation_indexing[int(starting_node_index) - 1]
        adjacency_list[starting_node] = []

        # If have adjacent nodes
        if part2.strip():
            adjacent_nodes = part2.split(',')
            for cur_adjacent_node in adjacent_nodes:
                cleaned_neighbor = cur_adjacent_node.strip()
                if not cleaned_neighbor.isdigit():
                   return f"Error: Invalid adjacent node '{cleaned_neighbor}' for node {starting_node}. Should be integers." 
                adjacency_list[starting_node].append(equation_indexing[int(cleaned_neighbor) - 1])


        # Formatting
        if len(adjacency_list[starting_node]) == 0:
            adjacency_list[starting_node] = [None]

    return adjacency_list



def get_gemini_adj_list(model):
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()
    total_text = ""

    for i, (cur_article_id, cur_article) in enumerate(article_ids.items()):
        if cur_article_id == "0907.2648":
            # Construct the HTML file path for the current article
            html_path = f'articles/{cur_article_id}.html'
            # Check if the HTML file exists
            if os.path.exists(html_path):
                # Read the content of the HTML file
                with open(f'articles/{cur_article_id}.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    
                equations, words_between_equations, equation_indexing = derivation_graph.extract_equations(html_content)

                equation_alttext = []
                total_text = words_between_equations[0]
                for i, cur_equation in enumerate(equation_indexing):
                    cur_alttext = ""
                    for j, cur_sub_equation in enumerate(equations[cur_equation]['equations']):
                        total_text += " " + cur_sub_equation['alttext']
                        cur_alttext += " " + cur_sub_equation['alttext']
                    total_text += " " + words_between_equations[i + 1]
                    equation_alttext.append(cur_alttext)
                
                prompt = "I have the following article that contains various mathematical equations: \n" + total_text 
                prompt += "\n From this article, I have extracted the list of equations, numbers as follows: \n"
                for i, cur_equation in enumerate(equation_alttext):
                    prompt += f"{str(i+1)}. {cur_equation}\n"
                prompt += "\n Analyze the context of the article to identify which equations are derived from each equation. Provide the output as a list and nothing else, with the format: w -> x, y, z;\n x -> h, t;\n ... If no equations are derived from a certain equation, return an empty list with the format: t ->;\n"

                raw_response = model.generate_content(prompt)
                text_response = ""
                for part in raw_response.parts:
                    text_response += part.text

                adjacency_list = parse_adjacency_list(text_response, equation_indexing)




    
    return ""
    # prompt = "Given "

    # raw_response = model.generate_content(prompt)
    # text_response = ""
    # for part in raw_response.parts:
    #     text_response += part.text
    # return text_response


if __name__ == '__main__':
    #  print(get_gemini_adj_list())
    print(parse_adjacency_list("1 ->;\n2 -> 3, 4;\n3 ->;\n4 -> 5;\n5 -> ;\n6 -> ;\n7 -> 8, 9, 10;\n8 -> ;\n9 -> ;\n10 ->;", ["S0.E1","S0.E2","S0.E3","S0.E4","S0.E5","S0.E6","S0.E7","S0.E8","S0.E9","S0.E10"]))