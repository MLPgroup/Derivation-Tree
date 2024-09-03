from old.pdfFindEq import eqno
from old.pdfFindEq import output

# Nodes and Links array
nodes = []
links = []

# Counters for third for-loop
initialEq = 1                                             
standardRange = 1                                         

for i in range(len(eqno)):                                  # Iterating through all the equations
    if (isinstance(eqno[i][0], str)):
        initialEq += 1
        continue
    for idx in range(eqno[int(i)][0]):                       # Scanning for possible edges ex. 1 to 3; 1 to 7
        temp = str(eqno[idx][0]) + ')'                       # Added only 1 parantheses because some initial parantheses are cut off
        if (initialEq > 1):                                  # Greater then 0 means there were sub equations ex. 1a, 1b
            initialEq -= 1                                   
            standardRange = initialEq                        # Decrement and update counter
        for j in range (eqno[i-standardRange][1]+1, eqno[i][1]-1):          # Iterating through the strings between each equation ex. 433 to 573; 573 to 643
            # test = output[j]                                               
            if temp in output[j]:                                           # If equation # is within lines; add edge
                links.append('{ source: \'' + str(eqno[idx][0]) + '\', target: \''+ str(eqno[i][0])+'\' }')                # Add edge from idx to i to links list

    initialEq = 1                                           # Reset Counter
    standardRange = 1                                       # Reset Counter
    nodes.append('{ name: \''+str(eqno[i][0])+'\' }')       # Add each node to nodes list      

# print('var nodes = ', nodes)        # Debugging
# print('var links = ', links)        # Debugging

# Printing nodes/links to txt
with open('data.txt', 'a') as f:
    f.write('var nodes = [\n')
    for i in range(len(nodes)):     # Couldnt directly print list so iterating through each string element then printing
        f.write(nodes[i]+',\n')
    f.write(']\n')
    f.write('var links = [\n')
    for i in range(len(links)):     # Couldnt directly print list so iterating through each string element then printing
        f.write(links[i]+',\n')
    f.write(']')


# TODO LIST: 

# connect text file to html
# get rid of repetitive nodes in link array
# Directed edges for D3 graphs (Or color gradient) (Or transparent Tree)
# Ideas for miscellaneous edges: if equations close together, add edge. Incorporating grammar (transition words), Seperate output array into paragraphs
# Ideas for seed equation: parse thorugh conclusion to find important equation, algorithm for most outgoing edges, match conclusion text, with rest of text to see familiarity
# Look for ways to parse Latex Equations