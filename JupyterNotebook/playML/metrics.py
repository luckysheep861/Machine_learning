import math
import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], \
           "the size of y_true must be equal to the size of y_pridict"
    return sum(y_true == y_predict) / len(y_predict)

def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
           "the size of y_true must be equal to the size of y_pridict"
    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    return math.sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
           "the size of y_true must be equal to the size of y_pridict"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    return 1 - (mean_squared_error(y_true, y_predict) / np.var(y_true))

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0)&(y_predict == 0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1)&(y_predict == 1))

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1)&(y_predict == 0))

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0)&(y_predict == 1))

def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true,y_predict), FP(y_true,y_predict)],
        [FN(y_true,y_predict), TP(y_true,y_predict)]
    ])

def precision_score(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FP(y_true, y_predict))
    except:
        return 0.0

def recall_score(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FN(y_true, y_predict))
    except:
        return 0.0

def f1_score(precession, recall):
    try:
        return 2 * precession * recall / (precession + recall)
    except:
        return 0.0

def TPR(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FN(y_true, y_predict))
    except:
        return 0.0

def FPR(y_true, y_predict):
    try:
        return FP(y_true, y_predict) / (FP(y_true, y_predict) + TN(y_true, y_predict))
    except:
        return 0.0