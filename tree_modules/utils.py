import numpy as np


def select_majority(labels):

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_majority = unique_labels[label_counts.argmax()]

    return label_majority