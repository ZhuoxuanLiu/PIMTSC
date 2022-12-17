import numpy as np


def accuracy_score(trues, preds):
    acc = np.sum(trues == preds) / len(trues)
    return acc