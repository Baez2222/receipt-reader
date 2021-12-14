def spell_check(textlist):
    word_graph = {}
    word_weight = {}
    
    # create graph
    for text in textlist: # text = tokenized address
        currWord = "graph start"
        for i in range(len(text) + 1): # i = word index in text
            
            # if i == len(text): # if end of text list, change current word to current index (instead of )
            #     currWord = text[-1]
                
                
            # existing graph node
            if currWord in word_graph: # if current word has already been seen before
                found = False
                for j in range(len(word_graph[currWord])):
                    if i == len(text):
                        word_graph[currWord][j][1] += 1
                        found = True
                        break
                    elif word_graph[currWord][j][0] == text[i] and word_graph[currWord][j][2] == i:
                        word_graph[currWord][j][1] += 1
                        found = True
                        break
                if not found and i != len(text):
                    word_graph[currWord].append([text[i], 1, i])
                elif not found and i == len(text):
                    word_graph[currWord].append(["graph end", 1, i])
            # new word
            else:
                if i== len(text):
                    word_graph[currWord] = [["graph end", 1, i]]
                    continue
                else:
                    # print(currWord, text[i])
                    word_graph[currWord] = [[text[i], 1, i]]
            
            # word weight dict
            if i != len(text):
                if text[i] in word_weight:
                        word_weight[text[i]] += 1
                else:
                    word_weight[text[i]] = 1
                
                currWord = text[i]
        
    # print(word_graph)
    # print("\n\n\n\n\n\n")
    # print(word_weight)
    
    # find path with largest weights
    currNode = "graph start"
    address = ""
    depth = 0
    while currNode != "graph end":
        # print(currNode)
        highCount = 0
        highWord = ""
        for word in word_graph[currNode]: # checks every outgoing edge of current node
            if word[0] == "graph end":
                currNode = "graph end"
                highWord = "graph end"
                break
            # print(word[0], word[2], depth)
            if word_weight[word[0]] > highCount and word[2] - depth < 2 and word[2] - depth >= 0:
                highCount = word_weight[word[0]]
                highWord = word[0]
                depth = word[2]
        currNode = highWord
        if currNode != "graph end":
            address += highWord + " "
    # print(address)
    return address