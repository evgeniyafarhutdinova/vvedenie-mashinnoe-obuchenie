import pandas as pd
import numpy as np
from sklearn.svm import SVC

def main():
    data = pd.read_csv('svm-data.csv', header=None)
    X = data.iloc[:,1:].values
    Y = data.iloc[:,0].values
    clf = SVC(C=100000.0, random_state=241)
    clf.fit(X, Y)
    print (clf.support_)

if __name__ == "__main__":
    main()