"""
Utils function for simlr-ad.
"""
import matlab.engine
import numpy as np

def compute_simlr(X, nclusters):
    """
    Compute SIMLR using matlab.

    Wrapper that calls the matlab function and processes the returning objects
    into numpy arrays.
    """

    eng = matlab.engine.start_matlab()
    (y, S, F, ydata, alpha) = eng.compute_simlr(matlab.double(X.tolist()),
                                                nclusters, nargout=5)
    y = np.array(y._data).reshape(y.size, order='F').squeeze()
    S = np.array(S._data).reshape(S.size, order='F').squeeze()
    F = np.array(F._data).reshape(F.size, order='F').squeeze()
    ydata = np.array(ydata._data).reshape(ydata.size, order='F').squeeze()
    alpha = np.array(alpha._data).reshape(alpha.size, order='F').squeeze()
    return y, S, F, ydata, alpha

def compute_cimlr(X, nclusters):
    """
    Compute SIMLR using matlab.

    Wrapper that calls the matlab function and processes the returning objects
    into numpy arrays.
    """

    eng = matlab.engine.start_matlab()
    (y, S, F, ydata, alphaK) = eng.compute_cimlr(matlab.double(X.tolist()),
                                                nclusters, nargout=5)
    y = np.array(y._data).reshape(y.size, order='F').squeeze()
    S = np.array(S._data).reshape(S.size, order='F').squeeze()
    F = np.array(F._data).reshape(F.size, order='F').squeeze()
    ydata = np.array(ydata._data).reshape(ydata.size, order='F').squeeze()
    alphaK = np.array(alphaK._data).reshape(alphaK.size, order='F').squeeze()
    return y, S, F, ydata, alphaK

def estimate_number_clusters(X):
    eng = matlab.engine.start_matlab()
    (K1, K2) = eng.Estimate_Number_of_Clusters_SIMLR(matlab.double(X.tolist()), matlab.double(list(range(2,15))), nargout=2)
    K1 = np.array(K1._data).reshape(K1.size, order='F').squeeze()
    K2 = np.array(K2._data).reshape(K2.size, order='F').squeeze()
    return K1, K2, list(range(2,15))


def estimate_number_clusters_cimlr(X):
    eng = matlab.engine.start_matlab()
    (K1, K2) = eng.number_kernels_cimlr(matlab.double(X.tolist()), matlab.double(list(range(2,15))), nargout=2)
    K1 = np.array(K1._data).reshape(K1.size, order='F').squeeze()
    K2 = np.array(K2._data).reshape(K2.size, order='F').squeeze()
    return K1, K2, list(range(2,15))

def feat_ranking(S, X):
    """
    Compute feature ranking on the clusters.

    Wrapper for matlab function SIMLR_feature_Ranking, that computes
    laplacian scores for subsets of the test and ranks the different features.
    """
    eng = matlab.engine.start_matlab()
    (aggR, pval) = eng.SIMLR_Feature_Ranking(matlab.double(S.tolist()),
                                             matlab.double(X.tolist()),
                                             nargout=2)
    aggR = np.array(aggR._data).reshape(aggR.size, order='F').squeeze()
    pval = np.array(pval._data).reshape(pval.size, order='F').squeeze()
    return aggR, pval