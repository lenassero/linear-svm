#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" This script implements Newton's method with backtracking line-search.
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

import numpy as np
import math

def NewtonStep(x, f, g, h):
    """ Compute the Newton step at x for the function f.

    Parameters
    ----------

    x: array, shape = [d]
    f: function
        Function of x to minimize.
    g: function
        Gradient of f (function of x).
    h: function
        Hessian of f (function of x).

    Returns
    -------

    newton_step: array, shape = [d]
        Newton step to apply to x.
    gap: float
        Estimated gap between f(xnew) and min(f)
    """
    g = g(x)
    h = h(x)
    h_inv = np.linalg.inv(h)
    lambda_ = (g.T.dot(h_inv.dot(g)))**(1/2)

    newton_step = -h_inv.dot(g)
    gap = 1/2*lambda_**2

    return newton_step, gap

def backTrackingLineSearch(x, step, f, g, A, b, alpha = 0.3, beta = 0.5):
    """ Compute the step size minimizing(t) f(x + t*step) with 
    backtracking line-search.

    Parameters
    ----------

    x: array, shape = [d]
    step: float
        Step at x in the descend method.
    f: function
        Function of x to minimize.
    g: function
        Gradient of f (function of x).
    alpha: float, optional (default = 0.3)
        "Fraction of the decrease in f predicted by linear extrapolation that 
        we will accept".
    beta: float, optional (default = 0.5)
        Factor to reduce t in the search.

    Returns
    -------

    step_size: float
        Step size to use for the next update (x := x + step_size*step).
    """
    # Initial step size
    step_size = 1

    # Number of inequalities
    m = b.shape[0]

    # First update
    xnew = x + step_size*step

    # Update step_size until xnew in dom(f)
    while np.sum(A.dot(xnew) < b) < m:
        step_size *= beta
        xnew = x + step_size*step

    # First evaluation
    y = f(xnew)
    z = f(x) + alpha*step_size*g(x).T.dot(step)

    while y > z:
        step_size *= beta
        xnew = x + step_size*step

        # Evaluation
        y = f(xnew)
        z = f(x) + alpha*step_size*g(x).T.dot(step)

    return step_size

def newtonLS(x0, f, g, h, tol, A, b, alpha = 0.3, beta = 0.5):
    """ Minimize the function f starting at x0 using Newton's method with
    backtracking line-search.

    Parameters
    ----------

    x0: array, shape = [d]
    f: function
        Function of x to minimize.
    g: function
        Gradient of f (function of x).
    h: function
        Hessian of f (function of x).
    tol: float
        Stopping criterion (gap between f(new) and min(f) < tol)
    alpha: float, optional (default = 0.3)
        "Fraction of the decrease in f predicted by linear extrapolation that 
        we will accept".
    beta: float, optional (default = 0.5)
        Factor to reduce t in the search.

    Returns
    -------

    xstar: array, shape = [d]
        Estimated minimum of f.
    xhist: float
        History of all Newton updates.
    """
    # First step
    newton_step, gap = NewtonStep(x0, f, g, h)
    step_size = backTrackingLineSearch(x0, newton_step, f, g, A, b, alpha, beta)
    x = x0 + step_size*newton_step
    xhist = [x]

    while gap > tol:
        newton_step, gap = NewtonStep(x, f, g, h)
        step_size = backTrackingLineSearch(x, newton_step, f, g, A, b, alpha, beta)
        x += step_size*newton_step
        xhist.append(x)

    xstar = x
    
    return xstar, xhist
