import pandas as pd
import numpy as np
import csv
import pickle as pk
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


def find_metrics(y_predict, y_proba, y_test):

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba)
    auPR = average_precision_score(y_test, y_proba)  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


test_set = ['TS49', 'TS88']

with open('./output_csvs/metrics_ps_s.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Sensitivity', 'Specificity', 'Balanced_acc', 'Accuracy', 'F1-score', 'MCC', 'AUC', 'auPR'])
    for t in test_set:
        file_path = f'./prev_paper/y_true_{t}'
        y_test = pd.read_csv(file_path, header=None).values

        file_path = f'./prev_paper/ps_s_y_proba_{t}.csv'
        y_proba = pd.read_csv(file_path, header=None).values

        y_pred = np.where(y_proba > 0.5, 1, 0)

        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, y_proba, y_test)

        print('Sensitivity : {0:.3f}'.format(sensitivity))
        print('Specificity : {0:.3f}'.format(specificity))
        print('Balanced_acc : {0:.3f}'.format(bal_acc))
        print('Accuracy : {0:.3f}'.format(acc))
        print('F1-score: {0:.3f}'.format(f1_score_1))
        print('MCC: {0:.3f}'.format(mcc))
        print('AUC: {0:.3f}'.format(auc))
        print('auPR: {0:.3f}'.format(auPR))

        writer.writerow([t, '{0:.3f}'.format(sensitivity), '{0:.3f}'.format(specificity), '{0:.3f}'.format(bal_acc),
                         '{0:.3f}'.format(acc), '{0:.3f}'.format(f1_score_1), '{0:.3f}'.format(mcc),
                         '{0:.3f}'.format(auc), '{0:.3f}'.format(auPR)])

