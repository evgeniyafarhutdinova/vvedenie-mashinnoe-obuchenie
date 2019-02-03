import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

def max_precision(length, recall, precision):
    max = 0
    for i in range(length):
        if recall[i] < 0.7:
            continue
        if precision[i] > max:
            max = precision[i]
    return max

def main():
    data = pd.read_csv('classification.csv', header=None)
    true = data.iloc[1:,0].values
    pred = data.iloc[1:,1].values
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(true)):
        if true[i] == '1':
            if pred[i] == '1':
                TP += 1
            else:
                FN += 1
        else:
            if pred[i] == '1':
                FP += 1
            else:
                TN += 1
    print('TP = ', TP)
    print('FP = ', FP)
    print('FN = ', FN)
    print('TN = ', TN)
    print()

    print('accuracy = ', accuracy_score(true, pred))
    print('precision = ', precision_score(true, pred, average='binary', pos_label='1'))
    print('recall = ', recall_score(true, pred, average='binary', pos_label='1'))
    print('f = ', f1_score(true, pred, average='binary', pos_label='1'))
    print()

    data_scores = pd.read_csv('scores.csv')
    data_scores.convert_objects(convert_numeric=True)
    y_true = data_scores.iloc[1:,0].values
    y_score_logreg = data_scores.iloc[1:,1].values
    y_score_svm =    data_scores.iloc[1:,2].values
    y_score_knn =    data_scores.iloc[1:,3].values
    y_score_tree =   data_scores.iloc[1:,4].values
    print('logreg = ', roc_auc_score(y_true, y_score_logreg))
    print('svm = ', roc_auc_score(y_true, y_score_svm))
    print('knn = ', roc_auc_score(y_true, y_score_knn))
    print('tree = ', roc_auc_score(y_true, y_score_tree))
    print()

    precision, recall, thresholds = precision_recall_curve(y_true, y_score_logreg)
    print('logreg = ', max_precision(len(recall), recall, precision))
    precision, recall, thresholds = precision_recall_curve(y_true, y_score_svm)
    print('svm = ', max_precision(len(recall), recall, precision))
    precision, recall, thresholds = precision_recall_curve(y_true, y_score_knn)
    print('knn = ', max_precision(len(recall), recall, precision))
    precision, recall, thresholds = precision_recall_curve(y_true, y_score_tree)
    print('tree = ', max_precision(len(recall), recall, precision))
if __name__ == "__main__":
    main()