import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    data_train = pd.read_csv('perceptron-train.csv', header=None)
    X_train = data_train.iloc[:,1:].values
    Y_train = data_train.iloc[:,0].values
    data_test = pd.read_csv('perceptron-test.csv')
    X_test = data_test.iloc[:,1:].values
    Y_test = data_test.iloc[:,0].values

    clf = Perceptron(random_state=241)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)

    not_scaled = accuracy_score(Y_test, predictions)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf2 = Perceptron(random_state=241)
    clf2.fit(X_train_scaled, Y_train)
    predictions2 = clf2.predict(X_test_scaled)

    scaled = accuracy_score(Y_test, predictions2)

    print(scaled - not_scaled)

if __name__ == "__main__":
    main()