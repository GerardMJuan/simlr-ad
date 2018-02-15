"""
Funcions to calculate statistics.

Funcions that do statistical tests and feature selection over
the clustering and the features.
"""

# Catch warning of deprecation
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.preprocessing import normalize


def compute_univariate_test(F_train, X_train, config, out_dir, feat_names):
    """
    Compute univariate test over features.

    Do feature selection using univariate test over each feature for each
    cluster. See which features are selected.
    """
    scores = []
    pval = []
    clusters = []
    for i in X_train.clusters.unique():
        selected = X_train[X_train.clusters == i]
        selected_rid = selected[["RID", "DX_bl"]]
        selected_rid = selected_rid[((selected_rid.DX_bl == 'CN') |
                                     (selected_rid.DX_bl == 'AD'))]
        selected_rid = selected_rid.merge(F_train, 'inner', on='RID')
        X = np.array(selected_rid[feat_names])
        Y = np.array(selected_rid["DX_bl"])
        Y = np.array([1.0 if x == 'AD' else -1.0 for x in Y])
        scores_i, pval_i = f_classif(X, Y)
        scores.append(scores_i)
        pval.append(scores_i)
        clusters.append(i)
    return normalize(scores), pval, clusters


def compute_randomizedlasso(F_train, X_train, config, out_dir, feat_names):
    """
    Compute RandomizedLasso feat selection.

    Do RandomizedLasso to select features over each cluster. Return selected
    features.
    """
    scores = []
    clusters = []
    for i in X_train.clusters.unique():
        selected = X_train[X_train.clusters == i]
        selected_rid = selected[["RID", "DX_bl"]]
        selected_rid = selected_rid[((selected_rid.DX_bl == 'CN') |
                                     (selected_rid.DX_bl == 'AD'))]
        selected_rid = selected_rid.merge(F_train, 'inner', on='RID')
        X = np.array(selected_rid[feat_names])
        Y = np.array(selected_rid["DX_bl"])
        Y = np.array([1.0 if x == 'AD' else -1.0 for x in Y])
        rl = RandomizedLasso(alpha='bic', n_resampling=500, fit_intercept=False,
                             sample_fraction=0.85, scaling=0.1,
                             random_state=1714)
        rl.fit(X, Y)
        scores.append(rl.scores_)
        clusters.append(i)
    return normalize(scores), clusters
