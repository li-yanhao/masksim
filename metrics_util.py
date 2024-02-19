
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, accuracy_score

import numpy as np


def auroc(real_preds, fake_preds):
    """ area under the ROC curve """
    y_score = np.concatenate([real_preds, fake_preds])
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def aucpr(real_preds, fake_preds):
    """ area under the precision-recall curve """
    y_score = np.concatenate([real_preds, fake_preds])
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def ap(real_preds, fake_preds):
    """ average precision score """
    y_score = np.concatenate([real_preds, fake_preds])
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])
    return average_precision_score(y_true, y_score)


def acc(real_preds, fake_preds):
    """ accuracy: (TP + TN) / all """
    thres = 0.5
    y_pred = np.concatenate([real_preds, fake_preds])
    y_pred[y_pred > thres] = 1
    y_pred[y_pred < thres] = 0
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    ACC = (TP + TN) / len(y_pred)
    return ACC


def acc_best(real_preds, fake_preds):
    """ accuracy with the best threshold """
    y_pred = np.concatenate([real_preds, fake_preds])
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])

    P = len(fake_preds)
    N = len(real_preds)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    FP = fpr * N
    TP = tpr * P
    # FN = P - TP
    TN = N - FP
    ACC = (TP + TN ) / len(y_pred)
    return np.max(ACC)


def mcc(real_preds, fake_preds):
    """ Matthews's correlation coefficient """
    thres = 0.5
    y_pred = np.concatenate([real_preds, fake_preds])
    y_pred[y_pred > thres] = 1
    y_pred[y_pred < thres] = 0
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])
    P = len(fake_preds)
    N = len(real_preds)
    # FP = fpr * N
    # TP = tpr * P
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = P - TP
    TN = N - FP
    MCC = (TP * TN - FP * FN) / (np.sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) ) + 1e-10)
    return MCC


def mcc_best(real_preds, fake_preds):
    """ Matthews's correlation coefficient with the best threshold """
    y_pred = np.concatenate([real_preds, fake_preds])
    y_true = np.concatenate([np.zeros(len(real_preds)), np.ones((len(fake_preds)))])

    P = len(fake_preds)
    N = len(real_preds)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    FP = fpr * N
    TP = tpr * P
    FN = P - TP
    TN = N - FP
    MCC = (TP * TN - FP * FN) / (np.sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-10)

    return np.max(MCC)
