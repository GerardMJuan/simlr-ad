"""
Utils functions as support
"""
import numpy as np


def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    """
    Use bisection to implement the simplex projection
    """
    lower = 0
    upper = np.max(v)
    current = np.inf

    for it in range(max_iter):
        if np.abs(current) / z < tau and current < 0:
            break

        theta = (upper + lower) / 2.0
        w = np.maximum(v - theta, 0)
        current = np.sum(w) - z
        if current <= 0:
            upper = theta
        else:
            lower = theta
    return w
