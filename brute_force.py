'''
Description: Python code to process mathematical articles in HTML format, identify equations links, and generate adjacency lists 
             representing derivation graphs.
'''


# Import Modules
import article_parser
from bs4 import BeautifulSoup
import os
from nltk.tokenize import sent_tokenize



# Words that mean no equation is present
filter_keywords = ['Fig', 'fig', 'FIG', 'Figure', 'FIGURE', 'figure', 'Lemma', 'LEMMA', 
                  'lemma', 'Theorem', 'THEOREM', 'theorem', 'Section', 'SECTION', 'section'
                  'Sec', 'SEC', 'sec', 'Table', 'TABLE', 'table', 'Ref', 'REF', 'ref', 
                  'Reference', 'REFERENCE', 'reference']



"""
getAdjList(equations, paragraph_breaks, words, extended_words)
Input: equations -- tuples of (Eq#, Idx# of Eq#)
       paragraph_breaks -- tuples of (Eq#, start of Paragraph interval for that specific eq#)
       words -- array of strings/words for HTML doc
       extended_words -- tuples of (Eq#, end interval; one sentence after)
Return: adj_list -- adjacency list extracted
Function: Iterates through given Mathematical document and sets edges between given equations
"""
def get_adj_list(equations, paragraph_breaks, words, extended_words):
    # Create adjacency list         
    adj_list = {}    

    # Iterate through equations
    for i in range(len(equations)):
        # If scanning through paragraph before first equation, skip since no prior equations for linkage
        if i == 0:
            continue
        # Scanning for possible edges
        for idx in range(i):
            # Current possible edge
            current_equation = equations[idx][0]
            # Iterating through the strings between start and actual equation
            for j in range (paragraph_breaks[i][1]+1, equations[i][1]-1):
                # Filter 
                if ((j >= 2) and (str(current_equation) == words[j]) and ('equationlink' in words[j-1]) and (not any(keyword in words[j-2] for keyword in filter_keywords))):
                    if equations[idx][0] not in adj_list:
                        adj_list[equations[idx][0]] = []
                    if equations[i][0] not in adj_list[equations[idx][0]]:
                        adj_list[equations[idx][0]].append(equations[i][0])
            # Iterating through the sentences between each equation
            for j in range (equations[i][1]+1, extended_words[i][1]-1):
                # Filter
                if ((j >= 2) and (str(current_equation) == words[j]) and ('equationlink' in words[j-1]) and (not any(keyword in words[j-2] for keyword in filter_keywords))):     
                    if equations[idx][0] not in adj_list:
                        adj_list[equations[idx][0]] = []
                    if equations[i][0] not in adj_list[equations[idx][0]]:
                        adj_list[equations[idx][0]].append(equations[i][0])

    # Return adjacency list
    return adj_list



"""
get_end_interval(equations, word_counts)
Inputs: equations -- tuples of (Eq#, Idx# of Eq#)
        word_counts -- array of number of words per sentence
Returns: extended_words -- list for holding the chunks of text after the equation
Function: Get extend range of text from end of equation to one sentence after
"""
def get_end_interval(equations, word_counts):
    # List for holding the chunks of text after the equation
    extended_words = []
    # Iterate through equations
    for idx, current_equation in enumerate(equations): 
        # Start of the portion of text AFTER the equation                             
        start_index = current_equation[1]
        # Counter for idx of wordCount array
        word_index = 0
        # Iterate through word_counts array until total words exceed current index (startIdx)
        while word_counts[word_index] < start_index:
            # Interval will go one more then necessary
            word_index +=1
        # Set end interval
        sentenceEndIdx = word_counts[word_index]
           # Append current index as end of section
        extended_words.append([str(equations[idx][0])+'end', sentenceEndIdx+10])

    # Return extended words
    return extended_words



"""
get_start_interval(equations, string_array)
Inputs: equations -- tuples of (Eq#, Idx# of Eq#)
        string_array -- array of words in article
Returns: paragraph_breaks -- list of initial paragraph interval index into (Eq #, idx #) output array
Function: Get paragraph intervals for each equation
"""
def get_start_interval(equations, string_array):
    paragraph_breaks = []                                     
    counter = 0
    # Placeholder for latest occurence of a paragraph break before equation
    temp = 0
    # Marker placed to locate paragraph breaks
    paragraph = 'parabreak'

    # Iterating through (Eq, idx number) pairs
    for i in range(len(equations)):
        # Iterating through index between previous equation and current equation
        for idx in range(counter, equations[i][1]-1):
            currWord = string_array[idx]
            # If there is a paragraph break marker, Set latest occurrence of paragraph break
            if paragraph == currWord:
                temp = idx
        # Append index to paragraph break list
        paragraph_breaks.append([(str(equations[i][0])+'start'), temp])
        # Set counter to start of next equation
        counter = equations[i][1]
        # Set latest occurrence of paragraph break to start of next equation
        temp = equations[i][1]
    
    # Return paragraph intervals
    return paragraph_breaks



"""
get_equation_tuples(string_array)
Inputs: string_array -- array of words in article
Returns: equations -- list of tuples (equation #, line number) 
Function: get tuples of (equation number, index number)
"""
def get_equation_tuples(string_array):
    equations = []
    count = 1
    # Checking for equations + line number
    for i in range(len(string_array)):
        # If there is a block equation
        if string_array[i] == 'mathmarker':
            equations.append([count, i+1])
            count += 1
        
    # Return tuples
    return equations



"""
get_sentence_count(text)
Inputs: text -- original text from article
Returns: word_count -- list with number of words for each sentence
Function: Get the number of words for each sentence
"""
def get_sentence_count(text):
    # Split String on sentences
    tokenized = sent_tokenize(text)

    # Keeps track of # of words in each sentence
    word_count = []
    # For Each sentence in the text:
    for sentence in tokenized:          
        # Split the sentence on spaces and count # of words
        total_word_count = len(sentence.split())     
        # Add current word count with word count of previous sentence
        if len(word_count) > 0:
            word_count.append(total_word_count + word_count[-1])          
        else:           
            word_count.append(total_word_count)

    # Return word count
    return word_count



"""
get_array_of_strings(text)
Inputs: text -- original text from article
Returns: list_of_strings -- list with all strings in text
Function: Convert entire text to an array of strings/words
"""
def get_array_of_strings(text):
    list_of_strings = []
    temp = ''

    # Converting to array of strings
    for i in range(len(text)):
        # Adding chars together until find a space
        temp += (text[i])
        if text[i] == ' ':
            list_of_strings.append(temp[:-1])
            temp = ''
            continue

    # Return list of each string
    return list_of_strings



"""
parse_html(html_path, cur_article_id)
Inputs: html_path -- path to current article's html
        cur_article_id -- current article's id
Returns: mathMl -- MathML for equation 
         text -- original text of article
         equation_ids -- equation ids for each equation
Function: Extract required information from article html
"""
def parse_html(html_path, cur_article_id):
    # Check if the HTML file exists
    if os.path.exists(html_path):
        # Read the content of the HTML file
        with open(f'articles/{cur_article_id}.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            mathMl = []

            # Find all td tags with rowspan= any int
            td_tags_with_rowspan_one = soup.find_all('td', {'rowspan': True})

            for td_tag in td_tags_with_rowspan_one:
                # Find the closest ancestor that is either a table or tbody tag
                ancestor_table_or_tbody = td_tag.find_parent(['table', 'tbody'])

                while ancestor_table_or_tbody:
                    # Create a new element with the insert text
                    marker = soup.new_tag("span", text='mathmarker', **{'class': 'mathmarker'})

                    if ancestor_table_or_tbody.get('id'):
                        ancestor_table_or_tbody.insert_before(marker)
                        mathMl.append(ancestor_table_or_tbody)
                        break
                    else:
                        # If id not found, go to the next ancestor
                        ancestor_table_or_tbody = ancestor_table_or_tbody.find_parent(['table', 'body'])
            
            # Replace MathML with the text "unicodeError"
            for script in soup(['math']):
                script.string = "unicodeError"

            # Get rid of annoying citations
            for script in soup(['cite']):
                script.extract()

            # Adding paragraph break markers (parabreak) before each paragraph
            for script in soup(['p']):                      
                if script.get('class') == ['ltx_p']:        
                    script.insert_before("parabreak")

            # Adding edge markers (edge) before each equation
            for script in soup(['a']):                          
                if script.get('class') == ['ltx_ref']:
                    script.insert_before("equationlink")
                
            # Check for elements with class "mathmarker" and skip processing them
            for script in soup.find_all(recursive=True):
                if script.get('class') == ['mathmarker']:
                    script.insert_before("mathmarker")

            # Get final processed text (including markers)
            text = soup.get_text(' ', strip=True)

            # Remove References OR Acknowledgments (Last) section
            text = (text.rsplit("References", 1))[0]
            text = text.split("Acknowledgments")[0]

            # Extracts all Block/Numbered Equation ID's from a Mathematical Text
            equation_ids = []
            for tag in mathMl:
                if tag.get('id'):
                    equation_ids.append(tag.get('id'))

            # Return parsing outputs
            return mathMl, text, equation_ids
    else:
        print(f"Error: The file {html_path} does not exist.")
        return None, None, None



"""
get_full_adj_list(old_adj_list, conversion)
Inputs: old_adj_list -- old adjacency list
        conversion -- list of conversions for each key in old adjacency list
Returns: new_adj_list -- newly formatted adjacency list
Function: Format adjacency list correctly
"""
def get_full_adj_list(old_adj_list, conversion):
    # Cleaned up adjacency list
    new_adj_list = {}

    # Iterate through equation ids
    for i, cur_eq in enumerate(conversion):
        new_adj_list[str(cur_eq)] = []
        # Format correctly
        if i + 1 in old_adj_list:
            for j, next_eq in enumerate(old_adj_list[i+1]):
                new_adj_list[str(cur_eq)].append(str(conversion[next_eq - 1]))
        else:
            new_adj_list[str(cur_eq)] = [None]

    # Return correctly formatted adjacency list
    return new_adj_list



"""
brute_force_algo()
Inputs: none
Returns: article_ids -- list of article ids used by algorithm
         true_adjacency_lists -- list of true adjacency lists
         predicted_adjacency_lists -- list of predicted adjacency lists
Function: Run all code to run the brute force algorithm
"""
def brute_force_algo():
    # Get a list of manually parsed article IDs
    articles = article_parser.get_manually_parsed_articles()

    article_ids = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    # Iterate through article IDs
    for i, (cur_article_id, cur_article) in enumerate(articles.items()):
        # Construct the HTML file path for the current article
        html_path = f'articles/{cur_article_id}.html'
        mathML, text, equation_ids = parse_html(html_path, cur_article_id)
        if text is None or mathML is None or equation_ids is None:
            continue
        word_count = get_sentence_count(text)
        string_array = get_array_of_strings(text)

        # Create Tuples of (Eq#, Idx#)
        equations = get_equation_tuples(string_array)
        # Start of paragraph interval per equation                 
        start = get_start_interval(equations, string_array)
        # End of paragraph interval per equation
        extended_words = get_end_interval(equations, word_count)

        # Get adjacency list
        adjList = get_adj_list(equations, start, string_array, extended_words)

        # Append article id
        article_ids.append(cur_article_id)
        # Clean up adjacency list for formatting and append
        predicted_adjacency_lists.append(get_full_adj_list(adjList, equation_ids))
        # Append tru adjacency list
        true_adjacency_lists.append(cur_article['Adjacency List'])
    
    # Return outputs of brute force algorithm
    return article_ids, true_adjacency_lists, predicted_adjacency_lists
