from PyPDF2 import PdfReader
from string import ascii_lowercase as alc

reader = PdfReader('Barnett10.pdf')

# Holds char array of pdf
output = []

# Extracting text data
for i in range(len(reader.pages)):
    page = reader.pages[i]
    output.append(page.extract_text())

# Combining all pages of Mathematical Text
text = str(output)

# Output for string array; temp for holding strings; idx for current index 
output = []
temp = ''
idx = 0

# Converting to string array format opposed to char
for i in range(len(text)):
    temp += (text[i])                   # Adding chars together until find a space
    if text[i] == ' ':                  # Once space is found,
        output.append(temp[:-1])        # Add string to output array 
        temp = ''
        continue
    # if "\\n" in temp:
    #     output.append(temp[:-2])
    #     temp = ''
    #     continue

# eqno array holds (equation #, line number) pair
eqno = []
inner_vec = []
idx = 1             # All equations start from 1
asc = 97            # Ascii for 'a'

# Checking for equations + 
for i in range(len(output)):
    temp = '(' + str(idx) + ')'                     # Equation Number
    tempascii = '(' + str(idx) + chr(asc) + ')'     # Equation Number w/ subequation
    nextTemp = '(' + str(idx+1) + ')'               # Next Equation Number 
    nextAscii = '(' + str(idx+1) + 'a' + ')'        # Next Equation Number w/ subequation
    if temp in output[i]:                       # Equation is regular
        inner_vec = [idx, i]
        eqno.append(inner_vec)
        idx += 1
        continue
    if tempascii in output[i]:                  # Equation has subequation
        inner_vec = [str(idx)+chr(asc), i]
        eqno.append(inner_vec)
        asc += 1
        continue
    if nextTemp in output[i]:           # Next equation no longer has a, b, c etc.
        inner_vec = [idx+1, i]
        eqno.append(inner_vec)
        idx += 2
        asc = 97
        continue
    if nextAscii in output[i]:          # Next equation moves onto next idx w/ subequation
        inner_vec = [str((idx+1))+'a', i]
        eqno.append(inner_vec)
        idx += 2
        asc = 98
        continue

# Insert Paragraph Breaks into the (Eq #, idx #) output array 

element = []
paraBreak = []                                  # New array with paragraph breaks
counter = 0                                     # Counter for current Word in PDF
temp = 0                                        # Placeholder for most recent paragraph break before equation
temp1 = 0                                       # Placeholder for index of space in current word
space = "\n"
for i in range(len(eqno)):                             # Iterating through (Eq, idx number) pairs
    for idx in range(counter, eqno[i][1]-2):     # Iterating through idx between previous Eq and current Eq
        currWord = output[idx]
        if space in currWord:                   # If there is a '\n' in the current word
            temp1 = currWord.index(space)       # Find index for it to check if next word is capitalized (If there is a paragraph break)
            isCapital = temp1+1
            currLetter = currWord[isCapital]
            if (currLetter.isupper()):
                temp = idx+1
    element.append(str(i+1)+'start')
    element.append(temp)
    paraBreak.append(element)
    element = []
    counter = eqno[i][1]
    temp = eqno[i][1]


print("Paragraph breaks: ", paraBreak)
print("No Paragraph breaks: ", eqno)


# print(eqno)
# j = 0
# for s in output:
#     if '(10)' in s:
#         print(j)
#     j = j+1
        

# print(eqno)                             # Debugging
# print(output[222])
# print(output[56])
# for i in range(len(output)):                # Debugging reading entire pdf
#     output[i] = output[i].encode('utf-8')
# print(output)