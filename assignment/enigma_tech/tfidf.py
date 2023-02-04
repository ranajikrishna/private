
import sys
import pdb
import math

import pandas as pd
import collections
import numpy as np


def tfidf(termFreq_doc: dict, idf: dict) -> dict:

    tfidf = dict.fromkeys(termFreq_doc.keys(),0)
    tfidf['norm'] = 0
    for key, value in termFreq_doc.items():
        tfidf[key] = value * idf[key]
        tfidf['norm'] += (value*idf[key])**2 

    l2_norm = np.sqrt(tfidf.pop('norm'))
    tfidf_norm = {k: v/l2_norm for k, v in tfidf.items()}
    return tfidf_norm


def inverse_doc_freq(word_list: list([dict]), bow: list([str])) -> dict:

    N = len(bow)
    l_bow = []
    for b in bow:
        l_bow += set(b)

    counter = dict(collections.Counter(l_bow))
    idf = dict.fromkeys(word_list,0)
    for w in word_list:
        idf[w] = np.log((1 + N)/(1 + counter[w]))+1

    return idf
    

def term_freq(input_list: list([str])) -> dict:
    freq = dict.fromkeys(input_list, 0)
    count = len(input_list)
    for term in input_list:
        freq[term] += 1/count
    return freq 


def main():
#    first_sentence = "Data Science is the sexiest job of the 21st century"
#    second_sentence = "machine learning is the key for data science"
    first_sentence = 'Game of Thrones is an amazing tv series!'
    second_sentence = 'Game of Thrones is the best tv series!'
    third_sentence = 'Game of Thrones is so great'

    first_list =  first_sentence.split(" ")
    second_list = second_sentence.split(" ")
    third_list = third_sentence.split(" ")
    first_list = list(map(lambda x: x.lower(), first_list))
    second_list = list(map(lambda x: x.lower(), second_list))
    third_list = list(map(lambda x: x.lower(), third_list))
    word_list = set(first_list).union(set(second_list)).union(set(third_list))

    termFreq_first = term_freq(first_list)
    termFreq_second = term_freq(second_list)
    termFreq_third = term_freq(third_list)
    termFreq = pd.DataFrame([termFreq_first,termFreq_second,\
                                                    termFreq_third]).fillna(0)

    idf = inverse_doc_freq(word_list,[first_list,second_list,third_list]) 
    tfidf_first = tfidf(termFreq_first, idf)
    tfidf_second = tfidf(termFreq_second, idf)
    tfidf_third = tfidf(termFreq_third, idf)
    result = pd.DataFrame([tfidf_first, tfidf_second, tfidf_third]).fillna(0)
    pdb.set_trace()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()


