
import numpy as np
import json as js
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pdb

def multinomial_reg(question_train, excerpt_train, topic_train, question_test,\
                                                    excerpt_test, topic_test):

    clf = Pipeline([('vect', CountVectorizer()) \
                    ,('clf', LogisticRegression( \
                                multi_class='multinomial' \
                                ,solver='lbfgs'\
                                ,penalty='l2' \
                                ,C=1.0))])

    clf.fit(excerpt_train, topic_train)
    predict=clf.predict(excerpt_test)

    print(sum(predict==np.array(topic_test))/len(topic_test))

    return 


def read_input():
    json = open('trainingdata.json','r')
    data = json.readlines()
    json.close()
    question_train = list()
    excerpt_train = list()
    topic_train = list()
    for line in data[1:-1]:
        question_train.append(js.loads(line)['question'])
        excerpt_train.append(js.loads(line)['excerpt'])
        topic_train.append(js.loads(line)['topic'])

    question_test = list()
    excerpt_test = list()
    topic_test = list()
    input_txt = open('testcases/input00.txt','r')
    input_data = input_txt.readlines()
    input_txt.close()
    output_txt = open('testcases/output00.txt','r')
    output_data = output_txt.readlines()
    output_txt.close()
    i = 1
    for i in range(1,len(input_data)-1):
        question_test.append(js.loads(input_data[i])['question'])
        excerpt_test.append(js.loads(input_data[i])['excerpt'])
        topic_test.append(output_data[i][0:-1])
        i+=1

    return question_train, excerpt_train, topic_train, question_test, \
                                                excerpt_test, topic_test

def main():
    question_train, excerpt_train, topic_train, question_test, excerpt_test, \
                                                    topic_test = read_input()
    multinomial_reg(question_train, excerpt_train, topic_train, \
                                    question_test, excerpt_test, topic_test)

    return 



if __name__ == '__main__':
    status = main()
