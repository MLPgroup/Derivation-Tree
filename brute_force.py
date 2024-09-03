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

flip_keywords = []



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

# ------------------------------ # Starting Paragraph Intervals ------------------------------
# Description: Outputs initial paragraph interval index into (Eq #, idx #) output array 
# @Param    eqno = Tuples of (Eq#, Idx# of Eq#)
#           output = Array of strings/words of original HTML doc
# --------------------------------------------------------------------------------------------
def startInterval(equations, output):
    paraBreak = []                                      # New array with paragraph breaks
    counter = 0                                         # Counter for current Word in PDF
    temp = 0                                            # Placeholder for latest occurence of a paragraph break before equation
    paragraph = 'parabreak'                             # Marker placed to locate paragraph breaks
    for i in range(len(equations)):                          # Iterating through (Eq, idx number) pairs
        for idx in range(counter, equations[i][1]-1):        # Iterating through idx between previous Eq and current Eq
            currWord = output[idx]
            if paragraph == currWord:                   # If there is a parabreak marker...
                temp = idx                              # Set latest occurence of paragraph break
        paraBreak.append([(str(equations[i][0])+'start'), temp])    # Append index to paragraph break list
        counter = equations[i][1]                            # Set counter to start of next equation
        temp = equations[i][1]                               # Set latest occurence of paragraph break to start of next equation
    return paraBreak

# ----------------------------------- # Tuples (Eq#, Idx#) -----------------------------------
# Description: Creating an array of tuples (equation #, line number) 
# @Param output = Array of strings/words of original HTML doc
# --------------------------------------------------------------------------------------------
def eqTuples(output):
    eqno = []
    count = 1
    # Checking for equations + line number
    for i in range(len(output)):
        if output[i] == 'mathmarker':          # There is a block equation
            eqno.append([count, i+1])                            # i+2 since i = mathequation, i+1 = equation location, i+2 = equation #
            count += 1
    return eqno

# ----------------------------------- # Words per Sentence -----------------------------------
# Description: Keeps track of # of words in each sentence; Use for para interval extension
# @Param text = Original text of HTML Document
# --------------------------------------------------------------------------------------------
def sentenceCount(text):
    # Split String on Sentences
    tokenized = sent_tokenize(text)
    # Debugging for printing entire text w/o references AND split into sentences
    # print('Text Split into Sentences: ', tokenized)

    wordCount = []                                                  # Keeps track of # of words in each sentence; Use for para interval extension
    for sentence in tokenized:                                      # For Each sentence in the text:
        totalWordCount = len(sentence.split())                      # Split the sentence on spaces and count # of words
        if len(wordCount) > 0:                                      # If sentence idx > 0,
            wordCount.append(totalWordCount+wordCount[-1])          # Add current word count with word count of setence previous
        else:           
            wordCount.append(totalWordCount)                        # Else, append normally
    return wordCount


# ----------------------------------- Array of Strings/Words -----------------------------------
# Description: Converting entire text to an array of strings/words
# @Param text = Original text of HTML Document
# ----------------------------------------------------------------------------------------------
def arrOfStrings(text):
    output = []
    temp = ''
    # Converting to array of strings
    for i in range(len(text)):
        temp += (text[i])                   # Adding chars together until find a space
        if text[i] == ' ':                  # Once space is found,
            output.append(temp[:-1])        # Add string to output array 
            temp = ''
            continue
    # print(len(output))
    return output


# ----------------------------- Block Equation Extraction -----------------------------
# Description: Extracts all Block/Numbered Equations from a Mathematical Text
# @Param url = url of Mathematical Document
# -------------------------------------------------------------------------------------
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
                script.string = "unicodeError"              # All block equations have unique string prior 

            # Get rid of annoying citations
            for script in soup(['cite']):
                script.extract()            # Removed

            # Adding paragraph break markers (parabreak) before each paragraph
            for script in soup(['p']):                      # For all the tags that have 'p'
                if script.get('class') == ['ltx_p']:        # If class tag is labelled with 'ltx_p'
                    script.insert_before("parabreak")       # Insert marker before each paragraph

            # Adding edge markers (edge) before each equation
            for script in soup(['a']):                          # For all the tags that have 'a'
                if script.get('class') == ['ltx_ref']:          # If class tag is labelled with 'ltx_ref'
                    script.insert_before("equationlink")        # Insert marker before each equation
                
            # Check for elements with class "mathmarker" and skip processing them
            for script in soup.find_all(recursive=True):
                if script.get('class') == ['mathmarker']:
                    script.insert_before("mathmarker")

            # Get final processed text (including markers)
            text = soup.get_text(' ', strip=True)  # Get text with some whitespace

            # Remove References OR Acknowledgments (Last) section
            text = (text.rsplit("References", 1))[0]
            text = text.split("Acknowledgments")[0]  # Split string at "Acknowledgments" and take only string before it

        # ----------------------------- Block Equation id Extraction -----------------------------
        # Description: Extracts all Block/Numbered Equation ID's from a Mathematical Text
        # @Param mathML = list of mathML equation elements
        # -------------------------------------------------------------------------------------
            math_ids = []
            for tag in mathMl:
                if tag.get('id'):
                    math_ids.append(tag.get('id'))

            return mathMl, text, math_ids


def get_full_adj_list(old_adj_list, conversion):

    new_adj_list = {}

    for i, cur_eq in enumerate(conversion):
        new_adj_list[str(cur_eq)] = []
        if i+1 in old_adj_list:
            for j, next_eq in enumerate(old_adj_list[i+1]):
                new_adj_list[str(cur_eq)].append(str(conversion[next_eq - 1]))
        else:
            new_adj_list[str(cur_eq)] = [None]

    return new_adj_list


def brute_force_algo():
    # Get a list of manually parsed article IDs
    articles = article_parser.get_manually_parsed_articles()

    article_ids = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    # Iterate through article IDs
    for i, (cur_article_id, cur_article) in enumerate(articles.items()):
        # Construct the HTML file path for the current article
        #if cur_article_id == '1409.0380':
        # if cur_article_id == '0907.2648':
        html_path = f'articles/{cur_article_id}.html'
        mathML, text, eqIds = parse_html(html_path, cur_article_id)
        wordCount = sentenceCount(text)
        stringArr = arrOfStrings(text)

        equations = eqTuples(stringArr)                 # Create Tuples of (Eq#, Idx#)
        start = startInterval(equations, stringArr)     # Start Paragraph interval per equation
        extended_words = get_end_interval(equations, wordCount)         # End Paragraph interval per equation

        adjList = get_adj_list(equations, start, stringArr, extended_words)

        article_ids.append(cur_article_id)
        predicted_adjacency_lists.append(get_full_adj_list(adjList, eqIds))
        true_adjacency_lists.append(cur_article['Adjacency List'])
    
    return article_ids, true_adjacency_lists, predicted_adjacency_lists
