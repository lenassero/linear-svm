#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" This script implements the coordinate descent algorithm for optimizing 
the SVM dual problem.
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

import numpy as np

from transform_svm import transform_svm_dual
from itertools import cycle 

def coordinate_descent_svm_dual(X, y, tau, Q, p, x_0, tol):
    """ Coordinate descent method for solving the SVM dual problem.

    Parameters
    ----------
    
    X: array, shape = [d+1, n]
        Observations with an offset.
    y: array, shape = [n, 1]
        Labels.
    tau: float
        Regularization parameter.
    Q: array, shape = [n, n]
        Matrix obtained after re-writing the SVM dual problem as a quadratic
        problem (1/2*xTQx + pTx)
    p: array, shape = [n, 1]
        Vector obtained after re-writing the SVM dual problem as a quadratic
        problem (1/2*xTQx + pTx)  
    x_0: array
        Stricly feasible starting point.
    tol: float
        Tolerance of the coordinate descent method:
        Stop when |f(xnew) - f(xold)| < tol.

    Returns
    -------

    x_sol: array
        Optimal solution.
    xhist: list (array)
        History of x.
    """
    # Number of inequality constraints (number of simpler minimization 
    # problems)
    n = X.shape[1]

    # Inequality constraints
    lower_bound = 0
    upper_bound = 1/(tau*n)

    # Quadratic function to optimize 
    f = lambda x: 1/2*np.dot(x, Q.dot(x)) + p.dot(x)

    # Initialization (feasible point for the inequality constraints)
    x = x_0

    # History
    xhist = [x_0]

    # Coordinate descent
    for i in cycle(range(n)):
        x_old = np.copy(x)
        x[i] = truncating_operator((1 - sum(2*x[k]*y[k]*X[:, k].dot(X[:, i]) 
            for k in range(n) if k != i)) / (2*y[i]*sum(X[:, i]**2)), lower_bound, upper_bound)
        xhist.append(x)
        if abs(f(x) - f(x_old)) < tol:
            x_sol = x
            break

    return x_sol, xhist

def truncating_operator(x, inf, sup):
    """ Returns x if inf <= x <= sup, inf if x < inf and sup if x > sup.

    Parameters
    ----------

    x: float
    inf: float
    sup: float

    Returns
    -------

    float

    """
    if inf <= x <= sup:
        return x
    elif x < inf:
        return inf
    elif x > sup:
        return sup

if __name__ == '__main__':
    print(truncating_operator(-5, 0, 10))
        
