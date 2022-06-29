# Enter your code here. Read input from STDIN. Print output to STDOUT
'''
This code does not work since the data cannot be injested.
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def svm_model(train_X, train_Y, test_X, test_label):
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVC(kernel='linear',C=1))
    ])
    param_grid = dict(clf__C=np.logspace(-4,1,6),
        clf__kernel=['rbf','linear']
    )
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring='accuracy')
    pipe.fit(train_X, train_Y)
    
    predict = pipe.predict(test_X)
    return tuple(zip(test_label, predict)) 


def read_input():
    txt = open('trainingdata_input.txt','r')
    data = txt.readlines()
    txt.close()
    
    N, F = map(int,input().split(' '))
    data = list()
    [data.append(input().split(' ')) for _ in range(0,N)]
    train_X = list()
    train_Y = list()
    for line in data:
        tmp = list()
        for item in line:
            tmp.append(item[item.find(':')+1:])
        train_X.append(tmp[2:])
        train_Y.append(np.array(tmp[1:2],dtype=np.float64))
        
    T = int(input())
    test_data = list()
    [test_data.append(input().split(' ')) for _ in range(0,T)]
    test_X = list()
    test_label = list()
    for line in test_data:
        tmp = list()
        for item in line:
            tmp.append(item[item.find(':')+1:])
        
        test_label.append(line[0])
        test_X.append(tmp[1:])
        
        
    return train_X, train_Y, test_X, test_label

def main():
    train_X, train_Y, test_X, test_label = read_input()    
    result = svm_model(train_X,train_Y, test_X, test_label)
    for i in result:
        print(i[0],end='')
        if(i[1]==1.0):
            print(' +1')
        else:
            print(' -1')
    return 

if __name__ == '__main__':
    status = main()

