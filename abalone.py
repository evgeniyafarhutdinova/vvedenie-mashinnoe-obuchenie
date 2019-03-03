import pandas as pd
from numpy import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


def main():
    data = pd.read_csv('abalone.csv')
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = {}
    for i in range(1, 51):
        clf = RandomForestRegressor(n_estimators=i, random_state=1)
        for train_index, test_index in cv.split(X, y):
            pass
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(X_train, y_train)
            scores[i] = r2_score(y_test, clf.predict(X_test))

    print(scores)

if __name__ == "__main__":
    main()