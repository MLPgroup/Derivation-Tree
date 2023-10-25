import numpy as np
#############################################################
# USE FOR LATER ONCE MATHML COMPONENTS HAVE BEEN TRANSLATED #
#############################################################

class Node:
    def __init__(self, v):
        self.data = v
        self.left = None
        self.right = None

# Function for MathML Tree Post-Order Traversal
def POT(Tree):
    if Tree == None:            # If None then Stop
        return
    POT(Tree.left)              # Recurse Left
    POT(Tree.right)             # Recurse Right
    print(Tree.data, end=' ')   # Print Node Data           

# Debug function that prints pretty Matrix
def printMatrix(Matrix):
    for row in Matrix:
        print(row, '\n')

# Helper function for LongestCommonSubstring that finds max value along with (x,y) coords
def findMax(Matrix):
    max = (0,0,0)
    for i in range(len(Matrix)):
        for j in range(len(Matrix[0])):
            if Matrix[i][j]> max[2]:
                max = (i,j,Matrix[i][j])
    return max

# Helper function for LongestCommonSubstring that eliminates longest common substring in matrix
def addNegOne(Matrix, max):
    for i in range(max[0]-max[2]+1, max[0]+1):
        for j in range(max[1]-max[2]+1, max[1]+1):
            Matrix[i][j] = -1


# DP solution to finding all common substrings (Starting from largest w/o duplicates)
# Taking in 2 equations, will create a matrix showing matching substring

def LongestCommonSubstring(equation1, equation2):

    # Matrix of size length of first equation X length of second equation
    Matrix = [ [0] * len(equation2) for i in range(len(equation1))]

    # Populating Matrix
    for i in range(len(equation1)):
        for j in range(len(equation2)):
            if equation1[i] == equation2[j]:
                if i == 0 or j == 0:
                    Matrix[i][j] = 1
                else:
                    Matrix[i][j] = Matrix[i-1][j-1]+1
            else:
                Matrix[i][j] = 0
    # printMatrix(Matrix)

    # Iterate through matrix and finding total substring matchings
    max = findMax(Matrix)
    totLen = 0
    while max[2] > 2:
        addNegOne(Matrix, max)
        printMatrix(Matrix)
        totLen += max
        max = findMax(Matrix)
        # printMatrix(Matrix)

    # Once Matrix has been iterated, calculate percentage of equation that was shared
    # Change current % after further trials
    if totLen/len(equation1) > 0.25:
        return True
    elif totLen/len(equation2) > 0.25:
        return True
    else:
        return False

# DP solution to finding all common substrings (Starting from largest w/o duplicates)
# Taking in 2 equations, will create a matrix showing matching substring
def LongestCommonSubstring(equation1, equation2):
    # Matrix of size length of first equation X length of second equation
    Matrix = [ [0] * len(equation2) for i in range(len(equation1))]
    # Populating Matrix
    for i in range(len(equation1)):
        for j in range(len(equation2)):
            if equation1[i] == equation2[j]:
                if i == 0 or j == 0:
                    Matrix[i][j] = 1
                else:
                    Matrix[i][j] = Matrix[i-1][j-1]+1
            else:
                Matrix[i][j] = 0
    # Finding max substring lengths
    # maxi holds max value in matrix, subLen arr holds all common substring lengths, brk boolean for checking when to break
    maxi = np.amax(Matrix)
    subLen = []
    brk = False
    while maxi > 2:
        # Double For Loop for finding substr with max length
        for i in range(len(equation1)):
            for j in range(len(equation2)):
                # Once found, iterate through area of matrix to see if occupied by any previous substring
                if Matrix[i][j] == maxi:
                    for x in range(i-maxi-1, i):
                        for y in range(j-maxi-1, j):
                            # If space is reserved by any other substring set values to all -1 AND BREAK !!!
                            if Matrix[x][y] == -1:
                                brk == True
                                break
                        # Break to get out of row for loop after iterating through substring area of matrix
                        if brk == True:
                            break
                    # If break boolean was never changed to True, that means no other substring occupied
                    # the current area. Then, add substring length to substring array
                    if brk == False:
                        subLen.append(maxi)
                    # Always set substring area to -1 since has been added to list/already occupied by another substring
                    for x in range(i-int(maxi)-1, i):
                        for y in range(j-int(maxi)-1, j):
                            Matrix[x][y] = -1
                    brk = True
                    break
                else:
                    continue
            if brk == True:
                break
        brk = False
        maxi = np.amax(Matrix)
    # Once Matrix has been iterated, calculate percentage of equation that was shared
    # Change current % after further trials
    if np.sum(subLen)/len(equation1) > 0.25:
        return True
    elif np.sum(subLen)/len(equation2) > 0.25:
        return True
    else:
        return False