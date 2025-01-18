import time
from collections import deque
from transformers import pipeline
import openai



# Global variable for rate limiting
api_call_times_queue_hf = deque()
api_call_times_queue_chatgpt = deque()

hf_token = ""

# Initialize pipelines for different Hugging Face models
huggingface_models = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "falcon": "tiiuae/falcon-7b-instruct"
}



def configure_chatgpt(api_key):
    openai.api_key = api_key



def configure_hf(input_hf_token):
    global hf_token
    hf_token = input_hf_token



# Parse adjacency list function
def parse_adjacency_list(text_response, equation_indexing):
    adjacency_list = {}
    lines = text_response.strip().split('\n')
    for cur_line in lines:
        cur_line = cur_line.rstrip(';').strip()
        if '->' not in cur_line:
            return f"Error: Line '{cur_line}' is not correctly formatted (missing '->')."
        part1, part2 = cur_line.split('->')
        starting_node_index = part1.strip()
        if not starting_node_index.isdigit():
            return f"Error: Invalid node '{starting_node_index}'. Nodes should be integers."
        starting_node = equation_indexing[int(starting_node_index) - 1]
        adjacency_list[starting_node] = []
        if part2.strip():
            adjacent_nodes = part2.split(',')
            for cur_adjacent_node in adjacent_nodes:
                cleaned_neighbor = cur_adjacent_node.strip()
                if not cleaned_neighbor.isdigit():
                    return f"Error: Invalid adjacent node '{cleaned_neighbor}' for node {starting_node}. Should be integers."
                adjacency_list[starting_node].append(equation_indexing[int(cleaned_neighbor) - 1])
        if len(adjacency_list[starting_node]) == 0:
            adjacency_list[starting_node] = [None]
    return adjacency_list



def get_llm_adj_list(model_name, equations, words_between_equations, equation_indexing):
    global api_call_times_queue_hf
    global hf_token
    print("here1.1")

    if model_name not in huggingface_models:
        return None, 1, f"Model {model_name} is not configured."

    model = pipeline("text-generation", model=huggingface_models[model_name], use_auth_token=hf_token, trust_remote_code=True)

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

    current_time = time.time()
    while api_call_times_queue_hf and current_time - api_call_times_queue_hf[0] > 59:
        api_call_times_queue_hf.popleft()
    if len(api_call_times_queue_hf) >= 15:
        time_to_wait = 59 - (current_time - api_call_times_queue_hf[0])
        if time_to_wait > 0 and time_to_wait <= 60:
            time.sleep(time_to_wait)

    try:
        raw_response = model(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
    except Exception as e:
        return None, 1, f"Error generating response: {str(e)}"

    print("here1.2")
    print(raw_response)

    api_call_times_queue_hf.append(time.time())
    adjacency_list = parse_adjacency_list(raw_response, equation_indexing)
    print(adjacency_list)

    if isinstance(adjacency_list, str):
        return adjacency_list, -1, raw_response
    elif isinstance(adjacency_list, dict):
        return adjacency_list, 0, "Good"
    else:
        return adjacency_list, 1, "Unknown"



def get_chatgpt_adj_list(equations, words_between_equations, equation_indexing, cur_article_id, model="gpt-3.5-turbo-0125", prompt_file="prompts.txt"):
    global api_call_times_queue_chatgpt

    # Format the equations and surrounding text into a prompt
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

    # Write the prompt to a file
    try:
        with open(prompt_file, 'a') as f:
            f.write("\n" + "="*80 + f"\nPrompt for Article {cur_article_id}:\n")
            f.write(prompt + "\n")
    except Exception as e:
        return None, 1, f"Error writing prompt to file: {str(e)}"

    # Skip the API call and return success message since prompts are saved
    return "Prompt saved to file", 0, "Good"


    # # Rate limiting: Check if 3 requests per minute limit is being exceeded
    # current_time = time.time()
    # while api_call_times_queue_chatgpt and current_time - api_call_times_queue_chatgpt[0] > 20:
    #     api_call_times_queue_chatgpt.popleft()
    # if len(api_call_times_queue_chatgpt) >= 3:
    #     time_to_wait = 20 - (current_time - api_call_times_queue_chatgpt[0])
    #     if time_to_wait > 0 and time_to_wait <= 20:
    #         time.sleep(time_to_wait)

    # try:
    #     # Make API call to OpenAI
    #     response = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": "You are an assistant that specializes in understanding and analyzing mathematical articles."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=1000,
    #         temperature=0.5
    #     )
    #     raw_response = response['choices'][0]['message']['content'].strip()
    #     #############
    #     print(response)
    #     print(raw_response)
    # except Exception as e:
    #     return None, 1, f"Error generating response: {str(e)}"

    # # Log the API call time
    # api_call_times_queue_chatgpt.append(time.time())

    # # Parse the raw response into an adjacency list
    # adjacency_list = parse_adjacency_list(raw_response, equation_indexing)
    # print(adjacency_list)

    # if isinstance(adjacency_list, str):
    #     return adjacency_list, -1, raw_response  # Error in parsing
    # elif isinstance(adjacency_list, dict):
    #     return adjacency_list, 0, "Good"
    # else:
    #     return adjacency_list, 1, "Unknown"
