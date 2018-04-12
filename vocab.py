import sys

def getwordcount(filepath):
    word_count = {}
    corpus = open(filepath, r) # Insert tokenised corpus here
    for line in corpus:
        for word in line:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
    return word_count
