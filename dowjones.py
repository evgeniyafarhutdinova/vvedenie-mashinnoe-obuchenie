import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def main():
    data = pd.read_csv('close_prices.csv')
    X = data.loc[:,'AXP':]
    pca = PCA(n_components=10)
    pca.fit(X)
    print(pca.explained_variance_ratio_)

    X_new = pca.transform(X)
    print(pca.components_[0])
    comp0_w = pd.Series(pca.components_[0])
    comp0_w_top = comp0_w.sort_values(ascending=False).head(1).index[0]
    company = X.columns[comp0_w_top]
    print(company)

    X_new_0 = X_new[:,0]

    data2 = pd.read_csv('djia_index.csv')
    dj = data2.iloc[:,1].values
    cor = np.corrcoef(X_new_0, dj)
    print(cor)

if __name__ == "__main__":
    main()