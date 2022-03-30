from sklearn import metrics
from config import classifier_config
import numpy as np


def cal_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    average = classifier_config['metrics_average']
    precision = metrics.precision_score(y_true, y_pred, average=average, labels=np.unique(y_pred), zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average=average, labels=np.unique(y_pred), zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, labels=np.unique(y_pred), zero_division=0)
    each_classes = metrics.classification_report(y_true, y_pred, output_dict=True, labels=np.unique(y_pred),
                                                 zero_division=0)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    return {'precision': precision, 'recall': recall, 'f1': f1}, each_classes, confusion_matrix
