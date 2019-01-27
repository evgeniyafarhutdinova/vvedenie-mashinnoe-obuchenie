import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

def main():
    wine_data = pd.read_csv(r"wine.data", header = None)
    X_not_scaled = wine_data.iloc[:,1:].values
    X = scale(X_not_scaled)
    Y = wine_data.iloc[:,0].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    neighbors_range = [x for x in range(1, 51)]
    for parameter in neighbors_range:
        scores = []
        neigh = KNeighborsClassifier(n_neighbors=parameter)
        scores = cross_val_score(estimator=neigh, X=X, y=Y, cv=kf)
        cv_scores.append(np.mean(scores))
    print(cv_scores)

    np_scores = np.fromiter(cv_scores, np.float)
    results = {}
    for x in range(len(neighbors_range)):
        results[neighbors_range[x]] = cv_scores[x]
    print(results)
    result = [neighbors_range[np_scores.argmax()], cv_scores[np_scores.argmax()]]
    print(result)

if __name__ == "__main__":
    main()