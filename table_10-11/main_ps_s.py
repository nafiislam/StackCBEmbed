import random

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
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


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
        random.seed(1)

        file = f'./prev_paper/ps_s_y_proba_{t}.csv'
        y_proba = pd.read_csv(file, header=None).values

        # 0.5 > 1, 0.5 < 0
        y_pred = np.where(y_proba > 0.5, 1, 0)

        all_concat = np.concatenate((y_pred.reshape(-1, 1), y_proba.reshape(-1, 1)), axis=1)

        global_sensitivity = []
        global_specificity = []
        global_bal_acc = []
        global_acc = []
        global_prec = []
        global_f1_score_1 = []
        global_mcc = []
        global_auc = []
        global_auPR = []

        file = f'./prev_paper/y_true_{t}'
        feature_y_Benchmark = pd.read_csv(file, header=None).values

        for i in range(0, 20):
            print(f'------------------------------Fold {i + 1}------------------------------')

            rus = RandomUnderSampler(random_state=i + 1)
            X, y = rus.fit_resample(all_concat, feature_y_Benchmark)

            c = Counter(y)
            print(c)

            print('X : ', X.shape)
            print('y : ', y.shape)

            y_pred = X[:, 0]
            y_proba = X[:, 1:]

            sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, y_proba, y)

            print('Sensitivity : {0:.3f}'.format(sensitivity))
            print('Specificity : {0:.3f}'.format(specificity))
            print('Balanced_acc : {0:.3f}'.format(bal_acc))
            print('Accuracy : {0:.3f}'.format(acc))
            print('F1-score: {0:.3f}'.format(f1_score_1))
            print('MCC: {0:.3f}'.format(mcc))
            print('AUC: {0:.3f}'.format(auc))
            print('auPR: {0:.3f}'.format(auPR))

            global_sensitivity.append(sensitivity)
            global_specificity.append(specificity)
            global_bal_acc.append(bal_acc)
            global_acc.append(acc)
            global_prec.append(prec)
            global_f1_score_1.append(f1_score_1)
            global_mcc.append(mcc)
            global_auc.append(auc)
            global_auPR.append(auPR)

        mean_sensitivity = np.mean(global_sensitivity)
        std_sensitive = np.std(global_sensitivity)
        mean_specificity = np.mean(global_specificity)
        std_specificity = np.std(global_specificity)
        mean_bal_acc = np.mean(global_bal_acc)
        std_bal_acc = np.std(global_bal_acc)
        mean_acc = np.mean(global_acc)
        std_acc = np.std(global_acc)
        mean_prec = np.mean(global_prec)
        std_prec = np.std(global_prec)
        mean_f1_score_1 = np.mean(global_f1_score_1)
        std_f1_score_1 = np.std(global_f1_score_1)
        mean_mcc = np.mean(global_mcc)
        std_mcc = np.std(global_mcc)
        mean_auc = np.mean(global_auc)
        std_auc = np.std(global_auc)
        mean_auPR = np.mean(global_auPR)
        std_auPR = np.std(global_auPR)

        mean_std_format = lambda mean, std: f'{mean:.3f} ± {std:.3f}'

        row = [
            mean_std_format(mean_sensitivity, std_sensitive),
            mean_std_format(mean_specificity, std_specificity),
            mean_std_format(mean_bal_acc, std_bal_acc),
            mean_std_format(mean_acc, std_acc),
            mean_std_format(mean_f1_score_1, std_f1_score_1),
            mean_std_format(mean_mcc, std_mcc),
            mean_std_format(mean_auc, std_auc),
            mean_std_format(mean_auPR, std_auPR)
        ]

        writer.writerow([t] + row)

