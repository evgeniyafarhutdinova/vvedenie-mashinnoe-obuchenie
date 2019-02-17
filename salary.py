import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

def main():
    # train
    train = pd.read_csv('salary-train.csv')

    train['FullDescription'] = train['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
    vectorizer = TfidfVectorizer(min_df=2)
    fullDescription = vectorizer.fit_transform(train['FullDescription'])

    train['LocationNormalized'].fillna('nan', inplace=True)
    train['ContractTime'].fillna('nan', inplace=True)
    enc = DictVectorizer()
    locationNormalized = enc.fit_transform(train[['LocationNormalized']].to_dict('records'))
    contractTime = enc.fit_transform(train[['ContractTime']].to_dict('records'))

    X = hstack((fullDescription, locationNormalized, contractTime))

    clf = Ridge(alpha=1, random_state=241)
    clf.fit(X, train['SalaryNormalized'])

    # test
    test = pd.read_csv('salary-test-mini.csv')

    test['FullDescription'] = test['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
    fullDescriptionTest = vectorizer.transform(test['FullDescription'])

    test['LocationNormalized'].fillna('nan', inplace=True)
    test['ContractTime'].fillna('nan', inplace=True)
    locationNormalizedTest = enc.transform(test[['LocationNormalized']].to_dict('records'))
    contractTimeTest = enc.transform(test[['ContractTime']].to_dict('records'))

    X_test = hstack((fullDescriptionTest, locationNormalizedTest, contractTimeTest))

    y_test = clf.predict(X_test)

    print(y_test)

if __name__ == "__main__":
    main()