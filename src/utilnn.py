import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import pearsonr


def accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    @param y_true, y_pred: np matrix, every line is a true/pred value array.
    """
    assert y_true.shape == y_pred.shape
    # Change labels to one-hot
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    # Calculate accuracy
    accuracy = sum(y_true == y_pred) / y_true.shape[0]
    return accuracy


def fscore(y_true, y_pred):
    """
    Calculate fscore.
    @param y_true, y_pred: np matrix, every line is a true/pred value array.
    """
    # Change labels to one-hot
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    fscore = f1_score(y_true, y_pred, average='macro')
    return fscore


def coef(y_true, y_pred):
    """
    Calculate coef.
    @param y_true, y_pred: np matrix, every line is a true/pred value array.
    """
    assert y_true.shape == y_pred.shape
    # Calculate coef one by one
    coefs = np.zeros(shape=(y_true.shape[0], 2))
    for i in range(coefs.shape[0]):
        coefs[i] = pearsonr(y_true[i], y_pred[i])
    # ... and return their average
    coef = np.average(coefs, axis=0)
    return coef
