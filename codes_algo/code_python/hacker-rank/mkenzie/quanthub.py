
import pdb
import sys
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,\
                                                                random_state=42)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    sc.fit(X_test)
    X_test_std = sc.transform(X_test)
    pdb.set_trace()

    

    return 



if __name__ == '__main__':
    status = main()
    sys.exit()
