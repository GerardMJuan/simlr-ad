"""Implements the LADCE baseline algorithm, with regression"""

import numpy as np
from sklearn import svm, preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
import random
import cvxpy as cvx


class LADCE_reg():
    """
    LACDE is an ensemble of regressors for AD in different age ranges.

    Parameters need to be validated.
    """

    def __init__(self, nclusters):
        """
        Create the class.

        nclusters is the number of clusters into which divide the
        different ages. C and k are parameters of the simple SVM classifier.
        classifiers and bootstrapping weights are data structures where the
        classifiers and weights will be stored.
        """
        self.fitted = False
        self.nclusters = nclusters
        self.classifiers = []
        self.bt_weights = [[] for x in range(nclusters)]

    def reg_cvxpy(self, X, Y):
        """
        Compute regresion in cvxpy.

        Use cvxpy to optimize a simple regression classifier,
        with the constraints that the weights need to be positive
        and to sum to 1, in order to do feature selection.
        """
        beta = cvx.Variable(X.shape[1])
        lambd = cvx.Parameter(sign="Positive")
        v = cvx.Variable()
        loss = cvx.norm((X * beta - v) - Y)**2
        reg = cvx.norm(beta, 2)**2
        m = X.shape[0]
        lambd.value = 0.1
        constraints = [0 <= beta]
        prob = cvx.Problem(cvx.Minimize(loss + lambd * reg), constraints)
        prob.solve()
        return beta.value, v.value

    def bootstrap(self, X, Y, age, nsamples=1000):
        """
        Bootstrap each classifier.

        Use resampling to boostrap all the classifiers to test their stability.
        """
        agerange = (91 - 54) / self.nclusters
        # train each classifier
        for i in range(0, self.nclusters):
            # Divide the data into the clusters
            l_age = 54 + agerange * i
            h_age = 54 + agerange * (i + 1)
            indexs = np.where((age > l_age) & (age < h_age))
            X_c = np.take(X, indexs, axis=0).squeeze()
            Y_c = np.take(Y, indexs, axis=0).squeeze()
            # Print the number of classes in each cluster
            print('Compute clusters between age ' + str(l_age) + ' and ' + str(h_age))
            print('Number of subjects: ' + str(len(Y_c)))
            # For each bootstrapping sample
            for j in range(nsamples):
                # sample the data
                X_sampled, Y_sampled = resample(X_c, Y_c)
                # Train the classifier
                weights, _ = self.reg_cvxpy(X_sampled, Y_sampled)
                # Save the classifier weights
                self.bt_weights[i].append(weights)

    def fit(self, X, Y, age):
        """
        Fit the algorithm to the data.

        X are the features of the samples, Y are the labels (DX, age).
        """
        agerange = (91 - 54) / self.nclusters

        # train each classifier
        for i in range(self.nclusters):
            # Divide the data into the clusters
            l_age = 54 + agerange * i
            h_age = 54 + agerange * (i + 1)
            indexs = np.where((age > l_age) & (age < h_age))
            X_c = np.take(X, indexs, axis=0).squeeze()
            Y_c = np.take(Y, indexs, axis=0).squeeze()
            # Print the number of classes in each cluster
            print('Compute clusters between age ' + str(l_age) + ' and ' + str(h_age))
            # Train the classifier
            weights, v = self.reg_cvxpy(X_c, Y_c)
            # Save the classifier
            self.classifiers.append((weights, v))
        self.fitted = True

    def predict(self, X, age, Y=None):
        """
        Return the scores of the features using the fitted model.

        X are the features of the samples.
        """
        if (not self.fitted):
            return "Need to fit the model first!"

        agerange = (91 - 54) / self.nclusters
        predictions = [0 for x in range(len(X))]
        for i in range(self.nclusters):
            # Divide the data into the clusters
            l_age = 54 + agerange * i
            h_age = 54 + agerange * (i + 1)
            indexs = np.where((age > l_age) & (age < h_age))
            X_p = np.take(X, indexs, axis=0)
            c_pred = X_p.dot(self.classifiers[i][0]) - self.classifiers[i][1]
            # Save the classifier
            for (i, p) in zip(indexs[0], np.array(c_pred)[0]):
                predictions[i] = p
        return predictions
