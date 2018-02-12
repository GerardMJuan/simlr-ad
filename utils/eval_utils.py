"""
Functions to evaluate classification performance.

These functions are useful to test for the peformance, classification-wise, of
the generated markers.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm


def compute_classification_results(pair, final_data,
                                   config, cv=False, test_data=None):
    """
    Compute binary classification results of a given marker.

    For the marker in final_data.scores, associated with the subjects,
    compute a series of classification scores following the procedure
    on the config file. Also, decide if we do crossvalidation or not.
    """
    # Do only binary classification with selected labels
    final_data = final_data[((final_data.DX == pair[0]) |
                             (final_data.DX == pair[1]))]

    # Extract the info
    X = final_data.pred.values
    Y = final_data.DX.values

    # Binarize the labels
    lb = preprocessing.LabelBinarizer()
    Y = lb.fit_transform(Y)

    if test_data is not None:
        # Do only binary classification with selected labels
        test_data = test_data[((test_data.DX == pair[0]) |
                               (test_data.DX == pair[1]))]

        # Extract the info
        X_test = test_data.pred.values
        Y_test = test_data.DX.values

        # Binarize the labels
        lb = preprocessing.LabelBinarizer()
        Y_test = lb.fit_transform(Y_test)
        X_test = lb.fit_transform(X_test)


#   # Initiate struct for results
    results = {
        'fold': [],
        "acc": [],
        "spe": [],
        "sen": [],
        "bacc": []
    }
    # Load configuration of the classification procedure
    if cv:
        nfolds = int(config['data_settings']['nfolds'])
        rd_seed = int(config['general']['random_seed'])
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True,
                              random_state=rd_seed)
        nfold = 0
        for train, test in skf.split(X, Y.ravel()):
            svc = svm.SVC(kernel='rbf', gamma=0.001, C=10)
            svc.fit(np.array(X[train]).reshape(-1, 1), Y[train].ravel())
            pred = svc.predict(X[test].reshape(-1, 1))
            # Compute metrics
            results = compute_metrics(Y[test].ravel(), pred, results)
            results['fold'].append(nfold)
            nfold = nfold + 1
        # save the scores
        results = pd.DataFrame(results)
        results.loc['mean'] = results.mean()
        print('Mean of results')
        print(results.loc['mean'])
    else:
        pred = X_test
        # Compute metrics
        results = compute_metrics(Y_test.ravel(), pred, results)
        results['fold'].append(1)
        results = pd.DataFrame(results)
        print(results)
    return results
    # results.to_csv(results_out)


def specificity_score(y_true, y_pred):
    """Only binary classification."""
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FP = CM[0][1]
    specificity = TN / (FP + TN)
    return specificity


def sensitivity_score(y_true, y_pred):
    """Only binary classification."""
    CM = confusion_matrix(y_true, y_pred)
    FN = CM[1][0]
    TP = CM[1][1]
    sensitivity = TP / (TP + FN)
    return sensitivity


def balanced_accuracy(y_true, y_pred):
    """Only binary classification."""
    sensitivity = sensitivity_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return (sensitivity + specificity) / 2


def compute_metrics(Y_test, Y_pred, problem_type):
    """
    Compute metrics for the predicted values.

    Y_test: original values to compare.
    Y_pred: predicted values.
    problem type: either regression or clasification.
    """
    if problem_type == "classification":
        return compute_metrics_classification(Y_test, Y_pred)
    else:
        print("Invalid type of problem!")


def compute_metrics_classification(y_true, y_pred):
    """
    Compute several metrics for classification.

    Computes accuracy, balanced accuracy, specifivity and sensitivity.
    Returns everything added to the results dictionary.
    """
    results = {
        "acc": [],
        "spe": [],
        "sen": [],
        "bacc": []
    }
    acc = accuracy_score(y_true, y_pred)
    spe = specificity_score(y_true, y_pred)
    sen = sensitivity_score(y_true, y_pred)
    bacc = balanced_accuracy(y_true, y_pred)
    results['acc'].append(acc)
    results['spe'].append(spe)
    results['sen'].append(sen)
    results['bacc'].append(bacc)
    return results


def svm_score(X_train, Y_train, X_test, Y_test):
    """
    Compute scores for classification task.

    X_train and X_test are the scores for the train and test set,
    Y_train and Y_test are the classes. Binary classification.
    """
    # Should make here a comprovation with the number of classes of Y.
    binary_test = (len(set(Y_train))) == 2 & (len(set(Y_test)) == 2)
    assert binary_test, "Need binary classes!"

    # Preprocess labels
    clf = svm.SVC(kernel='precomputed')
    clf.fit(X_train, Y_train)
    sc = clf.predict(X_test)
    b_acc = balanced_accuracy(X_test, sc)
    return b_acc
