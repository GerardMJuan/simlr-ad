"""Implements the LADCE baseline algorithm."""

import numpy as np
from sklearn import svm, preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
import random
import cvxpy as cvx


class LADCE():
    """
    LACDE is an ensemble of classifiers for AD in different age ranges.

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

    def svm_cvxpy(self, X, Y):
        """
        Compute svm in a cvxpy.

        Use cvxpy to optimize a linear svm classifier,
        with  the constraints that the weights need to be
        positive.
        """
        beta = cvx.Variable(X.shape[1])
        v = cvx.Variable()
        lambd = cvx.Parameter(sign="Positive")
        loss = cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(Y, X * beta - v)))
        reg = cvx.norm(beta, 2)
        m = X.shape[0]
        lambd.value = 0.1
        constraints = [0 <= beta, cvx.sum_entries(beta) == 1]
        prob = cvx.Problem(cvx.Minimize(loss / m + lambd * reg), constraints)
        prob.solve()
        return beta.value, v.value

    def balanced_subsample(self, x, y, subsample_size=1.0):
        """
        Do a balanced subsample of the class.

        NYI
        """
        xs = []
        ys = []
        classes = np.unique(y)
        min_elems = None
        size = subsample_size * np.shape(x)[0]
        # size = 150
        class_size = size / len(classes)
        # Divide the data by classes
        # For each class
        for c in classes:
            elems = x[(y == c)]
            # Select, at random, class_size elements
            for i in range(round(class_size)):
                xs.append(random.choice(elems))
                ys.append(c)

        return np.array(xs), np.array(ys)

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
            values, counts = np.unique(Y_c, return_counts=True)
            print(list(zip(values, counts)))
            # For each bootstrapping sample
            for j in range(nsamples):
                # sample the data
                X_sampled, Y_sampled = self.balanced_subsample(X_c, Y_c)
                # Train the classifier
                weights, _ = self.svm_cvxpy(X_sampled, Y_sampled)
                # Save the classifier weights
                self.bt_weights[i].append(weights)
                # del clf

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
            weights, v = self.svm_cvxpy(X_c, Y_c)

            # Print the number of classes in each cluster
            print('Compute clusters between age ' + str(l_age) + ' and ' + str(h_age))
            values, counts = np.unique(Y_c, return_counts=True)
            print(list(zip(values, counts)))
            # Train the classifier
            # Save the classifier
            self.classifiers.append((weights, v))
        self.fitted = True

    def predict(self, X, age):
        """
        Return the scores of the features using the fitted model.

        X are the features of the samples.
        """
        if (not self.fitted):
            return "Need to fit the model first!"

        agerange = (91 - 54) / self.nclusters
        predictions = [0 for x in range(len(X))]
        for i in range(0, self.nclusters):
            # Divide the data into the clusters
            l_age = 54 + agerange * i
            h_age = 54 + agerange * (i + 1)
            indexs = np.where((age > l_age) & (age < h_age))
            X_p = np.take(X, indexs, axis=0)
            c_pred = np.sign(X_p.dot(self.classifiers[i][0]) - self.classifiers[i][1])
            # Save the classifier
            for (i, p) in zip(indexs[0], np.array(c_pred)[0]):
                predictions[i] = p

        return predictions
