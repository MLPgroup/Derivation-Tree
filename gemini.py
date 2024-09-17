# Import modules
import time
from collections import deque



'''Model Configuration'''
# genai.configure(api_key=os.environ["API_KEY"])
# model = genai.GenerativeModel("gemini-1.5-flash")
'''Model Configuration'''


# Global variable for rate limiting
api_call_times_queue = deque()



"""
parse_adjacency_list(text_response, equation_indexing)
Input: text_response -- text response of Gemini LLM
       equation_indexing -- list storing the equation mathml tag for each index
Return: adjacency_list -- parsed adjacency list or error string
Function: Parse the text response from Gemini and construct the correctly formatted adjacency list
"""
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
        
        # Split input line into parts
        part1, part2 = cur_line.split('->')

        # Clean up
        starting_node_index = part1.strip()
        # Format checking
        if not starting_node_index.isdigit():
            return f"Error: Invalid node '{starting_node_index}'. Nodes should be integers."
        
        # Set up current starting node
        starting_node = equation_indexing[int(starting_node_index) - 1]
        adjacency_list[starting_node] = []

        # If have adjacent nodes
        if part2.strip():
            adjacent_nodes = part2.split(',')
            # Iterate through neighbors
            for cur_adjacent_node in adjacent_nodes:
                cleaned_neighbor = cur_adjacent_node.strip()
                # Format checking
                if not cleaned_neighbor.isdigit():
                   return f"Error: Invalid adjacent node '{cleaned_neighbor}' for node {starting_node}. Should be integers." 
                # Append correct neighbor to adjacency list
                adjacency_list[starting_node].append(equation_indexing[int(cleaned_neighbor) - 1])

        # Formatting
        if len(adjacency_list[starting_node]) == 0:
            adjacency_list[starting_node] = [None]

    # Return parsed adjacency list
    return adjacency_list



"""
get_gemini_adj_list(model, equations, words_between_equations, equation_indexing)
Input: model -- Gemini model to send API request to
       equations -- list storing the equations for current equation
       words_between_equations -- list of article text occurring between each equation
       equation_indexing -- ist storing the equation mathml tag for each index
Return: adjacency_list -- parsed adjacency list or error string
        error -- integer value if run correctly, 0 = run correctly, -1 = parsing error when parsing text response, 1 = other
        error_string -- string explaining current error
Function: Construct a prompt for the current article, ask the API for the response, and return the constructed adjacency list for the current article
"""
def get_gemini_adj_list(model, equations, words_between_equations, equation_indexing):
    global api_call_times_queue

    equation_alttext = []
    # Construct whole article with just text
    total_text = words_between_equations[0]
    # Add equations and rest of text
    for i, cur_equation in enumerate(equation_indexing):
        cur_alttext = ""
        # Add all parts of current equation
        for j, cur_sub_equation in enumerate(equations[cur_equation]['equations']):
            total_text += " " + cur_sub_equation['alttext']
            cur_alttext += " " + cur_sub_equation['alttext']
        total_text += " " + words_between_equations[i + 1]
        equation_alttext.append(cur_alttext)
    
    # Construct prompt
    prompt = "I have the following article that contains various mathematical equations: \n" + total_text 
    prompt += "\n From this article, I have extracted the list of equations, numbers as follows: \n"
    for i, cur_equation in enumerate(equation_alttext):
        prompt += f"{str(i+1)}. {cur_equation}\n"
    prompt += "\n Analyze the context of the article to identify which equations are derived from each equation. Provide the output as a list and nothing else, with the format: w -> x, y, z;\n x -> h, t;\n ... If no equations are derived from a certain equation, return an empty list with the format: t ->;\n"

    # Rate limit checking
    current_time = time.time()
    # Remove timestamps older than 60 seconds from the front of the queue
    while api_call_times_queue and current_time - api_call_times_queue[0] > 59:
        api_call_times_queue.popleft()
    # If there have been 15 or more calls in the last minute, wait
    if len(api_call_times_queue) >= 15:
        time_to_wait = 59 - (current_time - api_call_times_queue[0])
        if time_to_wait > 0 and time_to_wait <= 60:
            time.sleep(time_to_wait)

    # Get response from Gemini model
    raw_response = model.generate_content(prompt)

    # Enqueue the current time (i.e., add to the queue)
    current_time = time.time()
    api_call_times_queue.append(current_time)

    # Construct text response from Gemini model
    text_response = ""
    for part in raw_response.parts:
        text_response += part.text

    # Get adjacency list from gemini response
    adjacency_list = parse_adjacency_list(text_response, equation_indexing)

    # Check if response was parsed correctly
    if isinstance(adjacency_list, str):
        # Response parsed incorrectly due to wrong formatting
        return adjacency_list, -1, text_response
    elif isinstance(adjacency_list, dict):
        # Response parsed correctly
        return adjacency_list, 0, "Good"
    else:
        # Unknown error
        return adjacency_list, 1, "Unknown"