
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



import pdb;


def regression(corpus_train, train_dep, corpus_test):
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(corpus_train)
    x_test_counts = count_vect.transform(corpus_test)

    vectorizer = TfidfTransformer()
    X_train = vectorizer.fit_transform(x_train_counts)
    X_test = vectorizer.fit_transform(x_test_counts)
    
     # ********** Naive Bayes *********** #
#    model1 = GaussianNB()
#    model = MultinomialNB()
#
#    model.fit(X_train, train_dep)
#    predict = model.predict(X_test)
#    pdb.set_trace()
    # ************ #

     # ********** Multinomial Softmax *********** #
#    model = LogisticRegression(
#            multi_class='multinomial' \
#            ,solver='lbfgs' \
#            ,penalty='l2' \
#            ,C=1.0)
#    model.fit(X_train, train_dep)
#    predict = model.predict(X_test)
    # ************ #
   
    # ********** SVM  *********** #
    clf = make_pipeline(StandardScaler() \
                        ,SGDClassifier(max_iter=1000\
                        ,loss='hinge'\
                        ,penalty='l2'\
                        ,alpha=1e-3\
                        ,tol=1e-3))
    clf.fit(x_train_counts.toarray(), train_dep) 
    print(clf.predict(x_test_counts.toarray()))
    pdb.set_trace()

#    text_clf = Pipeline([('vect'\
#            ,CountVectorizer()\
#            ,TfidfTransformer()) \
#            ,('clf' \
#            ,SGDClassifier(loss='hinge'\
#            ,penalty='l2'\
#            ,alpha=1e-3\
##            ,n_iter=8\
#            ,random_state=42))])
#    pdb.set_trace()
#    text_clf = text_clf.fit(corpus_train,train_dep)
    
    #model1.fit(Y, train_dep)
    # ************ #

    
    #print(predict)
    return predict 

def features(x_train,Y_train,x_test):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    x_train_counts = count_vect.fit_transform(x_train)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    x_test_counts = count_vect.transform(x_test)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    y_train = np.array(Y_train)
    return x_train_tfidf, y_train, x_test_tfidf

def read_input():
    txt = open('trainingdata.txt','r')
    data = txt.readlines()
    txt.close()
    T = data[0]
    train_dep = list()
    corpus_train = list()
    [train_dep.append(line.split(' ')[0]) for line in data]
    train_dep = np.array(train_dep[1:],dtype=np.float64)
    [corpus_train.append(line[1:-1]) for line in data]
    txt = open('testdata.txt','r')
    data = txt.readlines()
    txt.close()
    dt = data[0] 
    corpus_test = list()
    [corpus_test.append(line[0:-1]) for line in data[1:]]
    return corpus_train[1:], train_dep, corpus_test

def main():
    corpus_train, train_dep, corpus_test = read_input()
    predict = regression(corpus_train, train_dep, corpus_test)
    return 


if __name__ == '__main__':
    status = main()
    
