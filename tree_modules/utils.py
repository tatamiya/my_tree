import numpy as np


def gini(y):
    _, counts = np.unique(y, return_counts=True)

    prob = counts / len(y)
    
    return 1 - (prob * prob).sum()


def select_majority(labels):

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_majority = unique_labels[label_counts.argmax()]

    return label_majority