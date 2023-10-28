from mathMLtoOP import *
import networkx as nx
import matplotlib.pyplot as plt

# Class for Adjacency List for Directed Graphs
class directGraph:
    # Dictionary for directed graph representation
    def __init__(self):
        self.graph = {}
    # Function for adding directed edge
    def addEdge(self, node1, node2):
        # create an empty list for a key node
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        self.graph[node1].append(node2)
    # Print graph
    def printGraph(self):
        print(self.graph)
    # Retrieve all directed edges
    def getEdges(self, node):
        if node in self.graph:
            return self.graph[node]
        else:
            return []

# ------------------------------------ # Seed Question --------------------------------------
# Description:  BFS function for removing repetitive edges. 
#               (Ex. a->b->c then edge a->c would be unecessary)
#               Return true if there is already a existing path. Else, false
# @Param    src: Source node (equation)
#           dest: Destination node (equation)
#           directedGraph: directGraph object
# --------------------------------------------------------------------------------------------

def bfs(src, dest, directedGraph):
    visited = [src]
    que = [src]
    while que:
        node = que.pop(0)
        if node == dest:
            return True
        for i in directedGraph.getEdges(node):
            if i not in visited:
                visited.append(i)
                que.append(i)
    return False

# ------------------------------------ # Seed Question --------------------------------------
# Description: Calculating Seed Equation; Finding number of incoming x outgoing nodes 
# @Param    directedGraph: directGraph object
# --------------------------------------------------------------------------------------------
def seedEq(directedGraph):
    max = 0
    eqNum = 'NULL'
    for key, value in directedGraph.graph.items():
        tempMax = 0
        tempMax += len(value)
        for key1, value1 in directedGraph.graph.items():
            if key1 == key:
                continue
            else:
                for i in value1:
                    if key == i:
                        tempMax += 1
        if max < tempMax:
            max = tempMax
            eqNum = key
    print('Seed Equation: ', eqNum)

# ---------------------------------- # Subtree Similarity ------------------------------------
# Description: Calculates Subtree Similarity between 2 given OPTrees 
# @Param    tree1 = OP Tree Root
#           tree2 = OP Tree Root
# --------------------------------------------------------------------------------------------
def partial_tree_match(tree1, tree2):
    def are_subtrees_similar(node1, node2):
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        if node1.value != node2.value:
            return False
        if len(node1.children) != len(node2.children):
            return False
        match = True
        for child1, child2 in zip(node1.children, node2.children):
            if not are_subtrees_similar(child1, child2):
                match = False
                break
        return match

    def dfs(node1, node2):
        similarity = 0
        if are_subtrees_similar(node1, node2):
            similarity = 1
        for child1 in node1.children:
            for child2 in node2.children:
                similarity += dfs(child1, child2)
        return similarity
    similarity_score = dfs(tree1, tree2)
    return similarity_score
 
# ------------------------------------ # End of Interval -------------------------------------
# Description: Graphing function; Iterates through given Mathematical document and draws edges
#              Between given equations
# @Param    eqno = Tuples of (Eq#, Idx# of Eq#)
#           paraBreak = Tuples of (Eq#, start of Paragraph interval for that specific eq#)
#           output = Array of strings/words for HTML doc
#           results = Array of mathML elements
#           exten = Tuples of (Eq#, end interval; one sentence after)
# --------------------------------------------------------------------------------------------
def derivationTree(eqno, paraBreak, output, results, exten):
    counter = 0                                                 # Counting number of elements between intervals
    adjList = directGraph()                                     # Create Adjacency List Object            
    G = nx.DiGraph()                                            # Create Directed Graph
    for i in range(len(eqno)):
        if eqno[i][0] == 1:                                         # If scanning through paragraph before first equation, skip since no prior equations for linkage
            continue
        edgeFlag = False                                            # Boolean to check if an edge has been added. If none for an equation, check cosine similarity with all equations before it
        for idx in range(eqno[i][0]-1):                                 # Scanning for possible edges ex. 1 to 3; 1 to 7 (-1 since not looking for current equation number)
            counter = 0                                                 # Counter for number of words between paragraphs/equations
            eqNum = str(eqno[idx][0])                                   # eqNum = current possible edge
            for j in range (paraBreak[i][1]+1, eqno[i][1]-1):           # Iterating through the strings between start and actual equation ex. 433 to 573; 573 to 643
                # counter += 1                                            # Increment word counter
                if (j >= 2 and eqNum in output[j]) and ('equationlink' in output[j-1]) and ('Fig' not in output[j-2]):         # If correct eq number is in curr element/ 'edgee' marker in previous element/ 'equationlink' is NOT in element before that                         
                    if bfs(eqno[idx][0], eqno[i][0], adjList) == False:         # If there is no path between the two edges,
                        edgeFlag = True                                         # Edge was added so true
                        adjList.addEdge(eqno[idx][0], eqno[i][0])               # Create an edge
                        G.add_edge(eqno[idx][0], eqno[i][0])                    # Edge from idx to i
            for j in range (eqno[i][1]+1, exten[i][1]):                 # Iterating through the strings between each equation ex. 433 to 573; 573 to 643
                if (j >= 2 and eqNum in output[j]) and ('equationlink' in output[j-1]) and ('Fig' not in output[j-2]):          # If correct eq number is in curr element/ 'edgee' marker in previous element/ 'equationlink' is NOT in element before that                         
                    if bfs(eqno[idx][0], eqno[i][0], adjList) == False:         # If there is no path between the two edges,
                        edgeFlag = True                                         # Edge was added so true
                        adjList.addEdge(eqno[idx][0], eqno[i][0])               # Create an edge
                        G.add_edge(eqno[idx][0], eqno[i][0])                    # Edge from idx to i
        # If no previous edges were added for an equation (node), look for cosine similarity. If greater then 0.5 (arbitrary similarity), add edge
        if edgeFlag == False:
            baseEquation = str(results[i])
            bE = toOpTree(baseEquation)                                     # change curr mathML element to string
            for idx in range(eqno[i][0]-1):                                     # Scanning for possible edges ex. 1 to 3; 1 to 7 (-1 since not looking for current equation number)
                compareEquation = str(results[idx])
                cE = toOpTree(compareEquation)                           # Change possible edge equation mathML to vector
                if partial_tree_match(bE, cE) >= 1:         # If similarity is greater then arbitrary percentage,
                    if bfs(eqno[idx][0], eqno[i][0], adjList) == False:
                        adjList.addEdge(eqno[idx][0], eqno[i][0])               # Create an edge
                        G.add_edge(eqno[idx][0], eqno[i][0])                    # Edge from idx to i
    # Draw graph and put onto png
    nx.draw_shell(G, with_labels = True)                                        # Taking graph G, add labels
    plt.savefig("DerivationTree.png")                                           # Output onto DerivationTree.png
    # seedEq(adjList)
    # Debugging 
    # adjList.printGraph()
    return adjList

# TODO LIST:
#               - Testing program on multiple Mathematical Documents
#               - Overleaf
#               - Push all code to MLP Repo