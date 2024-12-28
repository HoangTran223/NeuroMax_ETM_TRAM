import numpy as np
from collections import defaultdict
from sklearn import metrics


def purity_score(y_true, y_pred):
    print(f"Shape of y_true: {y_true.shape}")
    print(f"Shape of y_pred: {y_pred.shape}")

    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, preds):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results


def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    print(f"Shape of theta: {theta.shape}")
    print(f"Shape of labels: {labels.shape}")
    return clustering_metric(labels, preds)
    
    return clustering_metric(labels, preds)
