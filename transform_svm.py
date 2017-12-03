#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" This scripts implements functions to transform the SVM primal and dual 
problems as particular instances of the quadratic optimization problem: 

minimize(x) 1/2*xTQx + pTx 
st. Ax <= b
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

import numpy as np

def transform_svm_primal(tau, X, y):
    """ Write the SVM primal problem as a quadratic problem.

    Parameters
    ----------
    
    tau: float
        Regularization parameter.
    X: array, shape = [d+1, n]
        Observations with an offset.
    y: array, shape = [n, 1]
        Labels.

    Returns
    -------

    Q: array, shape = [d+1+n, d+1+n]
    p: array, shape = [d+1+n, 1]
    A: array, shape = [2n, d+1+n]
    b: array, shape = [2n, 1]
    """
    # Features' dimension (d+1)
    d_ = X.shape[0]
    d = d_ - 1

    # Number of observations
    n = X.shape[1]

    # Multiply each observation (xi) by its label (yi)
    X_ = X*y

    # Shape (d+1+n, d+1+n)
    Q = np.zeros((d_+n, d_+n))
    Q[:d, :d] = np.identity(d)

    # Shape (d+1+n,)
    p = np.zeros(d_+n)
    p[d_:] = 1/(tau*n)

    # Shape (2*n, d+1+n)
    A = np.zeros((2*n, d_+n))
    A[:n, :d_] = -X_.T
    A[:n, d_:] = np.diag([-1]*n)
    A[n:, d_:] = np.diag([-1]*n)

    # Shape (2*n, )
    b = np.zeros(2*n)
    b[:n] = -1

    return Q, p, A, b

def transform_svm_dual(tau, X, y):
    """ Write the SVM dual problem as a quadratic problem.

    Parameters
    ----------
    
    tau: float
        Regularization parameter.
    X: array, shape = [d+1, n]
        Observations with an offset.
    y: array, shape = [n, 1]
        Labels.

    Returns
    -------

    Q: array, shape = [n, n]
    p: array, shape = [n, 1]
    A: array, shape = [2n, n]
    b: array, shape = [2n, 1]
    """
    # Number of observations
    n = X.shape[1]

    # Multiply each observation (xi) by its label (yi)
    X_ = X*y

    # Shape (n, n)
    Q = X_.T.dot(X_)

    # Shape (n, )
    p = -np.ones(n)

    # Shape (2*n, n)
    A = np.zeros((2*n, n))
    A[:n, :] = np.identity(n)
    A[n:, :] = -np.identity(n)

    # Shape (2*n, )
    b = np.zeros(2*n)
    b[:n] = 1/(tau*n)

    return Q, p, A, b