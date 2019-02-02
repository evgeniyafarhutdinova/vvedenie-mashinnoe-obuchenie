import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def main():
    newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space'])
    vectorizer = TfidfVectorizer()
    X = newsgroups.target
    y = newsgroups.data
    X_train = vectorizer.fit_transform(y)

    grid = {'C': np.power(10.0, np.arange(-5, 5))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X_train, X)

    clf.set_params(**gs.best_params_)
    clf.fit(X_train, X)

    feature_names = np.asarray(vectorizer.get_feature_names())
    top10 = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
    print(feature_names[top10].tolist())

if __name__ == "__main__":
    main()