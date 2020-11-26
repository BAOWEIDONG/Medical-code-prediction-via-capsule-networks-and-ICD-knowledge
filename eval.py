import pandas as pd
import numpy as np

from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys

from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from sklearn.metrics import *

# from constants import *

def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    names = ['acc', 'prec', 'rec', 'f1']

    # macro
    macro = all_macro(yhat, y)

    # micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + '_macro': macro[i] for i in range(len(macro))}

    metrics.update({names[i] + '_micro': micro[i] for i in range(len(micro))})

    # AUC and @k
    if yhat_raw is not None and calc_auc:
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2 * (prec_at_k * rec_at_k) / (prec_at_k + rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)

# Axis = 0 represents label level whereas axis = 1 represents instance level

def union_size(yhat, y, axis):
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) # This calculates true positives (1, 1) at a label level
    den = union_size(yhat, y, 0) + 1e-10 # This calculates all (1, 1), (1, 0), (0, 1) examples. +1e-10 for numerical stability.
    return np.mean(num/den)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) # TP
    den = yhat.sum(axis=0) + 1e-10 # All the ones in yhat i.e. TP + FP
    return np.mean(num/den)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0)
    den = y.sum(axis=0) + 1e-10 # All the ones in y i.e. TP + FN
    return np.mean(num/den)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)

def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)

def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2*(prec*rec)/(prec+rec)
    return f1


def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

def sk_micro_f1(y_pred_label,y_true_label):
    test_acc = accuracy_score(y_true_label, y_pred_label)
    test_precision = precision_score(y_true_label,y_pred_label,average='micro')
    test_recall = recall_score(y_true_label,y_pred_label,average='micro')
    test_f1 = f1_score(y_true_label, y_pred_label, average='micro')
    #print("micro precision: ",test_precision,"micro recall: ",test_recall,"micro f1: ",test_f1,)
    print("micro precision: %.4f, micro recall: %.4f, micro f1: %.4f." % (test_precision,test_recall,test_f1))
    return test_f1

def sk_macro_f1(y_pred_label,y_true_label):
    test_acc = accuracy_score(y_true_label, y_pred_label)
    test_precision = precision_score(y_true_label,y_pred_label,average='macro')
    test_recall = recall_score(y_true_label,y_pred_label,average='macro')
    test_f1 = f1_score(y_true_label, y_pred_label, average='macro')
    #print("macro precision: ",test_precision,"macro recall: ",test_recall,"macro f1: ",test_f1,)
    print("macro precision: %.4f, macro recall: %.4f, macro f1: %.4f." % (test_precision,test_recall,test_f1))
    return test_f1

if __name__ == '__main__':

    # yhat 预测y
    # y 真实y
    # yhat_raw 预测y值
    import pickle
    yhat = pickle.load(open('test_pred_y_onehot.pkl','rb'))
    y = pickle.load(open('test_y.pkl','rb'))
    yhat_raw = pickle.load(open('test_pred_y.pkl','rb'))
    yhat = np.array(yhat)
    y = np.array(y)
    yhat_raw = np.array(yhat_raw)
    metrics = all_metrics(yhat, y, k=8, yhat_raw=yhat_raw, calc_auc=True)
    sk_micro_f1(yhat,y)
    sk_macro_f1(yhat,y)
    print(metrics)
