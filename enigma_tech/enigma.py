"""
Goldilocks TF-IDF

In the course of your research, you make a surprising discovery: the most important words are neither the most frequent nor the least frequent, but the ones of average frequency. We are seeing so much signal in that which is in the middle that we need a modified TF-IDF algorithm that treats words of average occurrence as the most important and both very frequent and very infrequent words as the least important. You need to fit it to a corpus because your use case requires you to use this on an incoming stream of documents.

Fill in this class that encapsulates your discoveries. We've included instructions with each method to scope the functionality. If you can think of ways to test out your code, also feel free to include it here.

You can assume:
    * Your training corpus is the provided Goldilocks text at '/home/coderpad/data/goldilocks.txt'
    * This environment comes equipped with Anaconda libraries like Numpy and Scipy

IMPORTANT: You can feel free to look up any concepts or syntax you like, but please limit wholesale copying of code.
"""


import re
import pdb
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix


class GoldilocksTFIDF(object):
    def __init__(self):
        """Fill in initialization, if needed"""
        self.idf = None
    
    @staticmethod
    def preprocess_corpus(text):
        """
        The input text is an unstructured text file. Define a document
        as a paragraph delimited by two newlines.
        
        For text processing, convert,
            '"This porridge is too hot!" she exclaimed.'
        into,
            'this porridge is too hot she exclaimed'
        
        """
        alphanum_delim = re.sub(r"[^a-z0-9 \n]", " ", text.lower())   
        corpus = [t.split() for t in alphanum_delim.split("\n\n")]
        return corpus
    
    def _set_vocabulary(self, corpus):
        vocabulary = sorted(set([term for doc in corpus for term in doc]))
        self.vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        return
    
    def fit(self, text):
        """Insert fit logic here and populate self.idf"""
        corpus = self.preprocess_corpus(text)
        self._set_vocabulary(corpus)
        
        wordbook = dict.fromkeys(self.vocabulary.keys(),0)
        doc_unq = []
        for word_list in corpus:
            doc_unq += set(word_list)
            
        for term in wordbook.keys():
            wordbook[term] = sum(np.array(doc_unq)==term)
            
        word_mean = np.mean(list(wordbook.values()))
        word_std = np.std(list(wordbook.values()))
        zidf = dict.fromkeys(wordbook.keys(),0)
        for key,value in wordbook.items():
            zidf[key] = 1 - (abs(value-word_mean)/word_std)
        
        self.idf =  zidf
        return
    
    def transform(self, text):
        """You're given a new input text that resembles the corpus.
        What's your output?"""
        corpus = self.preprocess_corpus(text)
        result = np.zeros((len(corpus), len(self.vocabulary)))
        for i, doc in enumerate(corpus):
            term_freq = Counter(doc)
            for term in set(doc):
                idf_idx = self.vocabulary[term]
                result[i, idf_idx] = term_freq[term] * self.idf[term]
        return result


with open("/home/coderpad/data/goldilocks.txt", "r") as f:
    # load and view the data
    text = f.read()
    gtfidf = GoldilocksTFIDF()
    print(text)
    print(GoldilocksTFIDF.preprocess_corpus(text))
    
    #uncomment this when ready to test
    gtfidf.fit(text)
    print(csr_matrix(gtfidf.transform(text)))
