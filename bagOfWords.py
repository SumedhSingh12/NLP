import nltk
import numpy as np
import heapq
import re

#Insert the text between the triple quotes
paragraph = ''' '''

'''Normalising the text by first converting all the characters to lower case
   Then removing all the punctuations
   Removing all the extra spaces'''
   
for sentence in paragraph:
    sentence = sentence.lower()
    sentence = re.sub(r'\W', " ", sentence)
    sentence = re.sub(r'\s+', " ", sentence)
    
'''Now, creating a dictionary of all the words with their counts'''

word_to_count = {}
for sentence in paragraph:
    for word in nltk.word_tokenize(sentence):
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

'''The number of frequent words to be considered for the model is to 100 now. It can be changed by changing the parameter in the line below'''

frequent_words = heapq.nlargest(100, word_to_count, key = word_to_count.get)
big_bag = []
for sentence in paragraph:
    sentence_bag = []
    for word in frequent_words:
        if word in nltk.word_tokenize(sentence):
            sentence_bag.append(1)
        else:
            sentence_bag.append(0)
    big_bag.append(sentence_bag)

big_bag = np.asarray(big_bag)
