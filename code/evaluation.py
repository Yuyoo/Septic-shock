# -*- coding: utf-8 -*-
# @Time    : 2017/12/17 12:47
# @Author  : Xiaofeifei
# @File    : evaluation.py

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools


def auc(y_true, y_pred):
    y_pred = np.squeeze(np.reshape(y_pred, [-1, 1]))
    y_true = np.squeeze(np.reshape(y_true, [-1, 1]))

    return roc_auc_score(y_true, y_pred)


def plot_roc(y_true, y_pred, title='Receiver Operating Characteristic'):
    y_pred = np.squeeze(np.reshape(y_pred, [-1, 1]))
    y_true = np.squeeze(np.reshape(y_true, [-1, 1]))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([-0.01, 1.01])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(title)
    plt.show()


"""
def average_precision(y_true, y_pred):
    y_pred = np.squeeze(np.reshape(y_pred, [-1, 1]))
    y_true = np.squeeze(np.reshape(y_true, [-1, 1]))
    average_precision_ = average_precision_score(y_true, y_pred)    # 计算平均准确率
    return average_precision_
"""


def precision_recall(y_true, y_pred):
    y_pred = np.squeeze(np.reshape(y_pred, [-1, 1]))
    y_true = np.squeeze(np.reshape(y_true, [-1, 1]))
    cm = confusion_matrix(y_true, y_pred)
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    return precision, recall


def plot_confusion_matric(y_true, y_pred, classes, normalize=False, title='Confusion matrix'):
    y_pred = np.squeeze(np.reshape(y_pred, [-1, 1]))
    y_true = np.squeeze(np.reshape(y_true, [-1, 1]))
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))  # 伪坐标轴
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
