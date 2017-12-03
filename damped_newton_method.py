#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" This script implements the damped Newton algorithm.
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

import numpy as np

def dampedNewtonStep(x, f, g, h):
    """ Compute the damped Newton step at x for the function f.

    Parameters
    ----------

    x: array, shape = [d]
    f: function
        Function of x to minimize.
    g: function
        Function of x of the gradient of f.
    h: function
        Function of x of the hessian of f.

    Returns
    -------

    xnew: array, shape = [d]
        Update: value of x after applying the step.
    gap: float
        Estimated gap between f(xnew) and min(f)
    """
    g = g(x)
    h = h(x)
    h_inv = np.linalg.inv(h)
    lambda_ = (g.T.dot(h_inv.dot(g)))**(1/2)

    xnew = x - (1/(1+lambda_))*h_inv.dot(g)
    gap = 1/2*lambda_**2

    return xnew, gap

def dampedNewton(x0, f, g, h, tol):
    """ Minimize the function f starting at x0 using the damped Newton 
    algorithm.

    Parameters
    ----------

    x0: array, shape = [d]
    f: function
        Function of x to minimize.
    g: function
        Function of x of the gradient of f.
    h: function
        Function of x of the hessian of f.
    tol: float
        Stopping criterion (gap between f(new) and min(f) < tol)

    Returns
    -------

    xstar: array, shape = [d]
        Estimated minimum of f.
    xhist: float
        History of all damped Newton updates.
    """
    # First step
    x, gap = dampedNewtonStep(x0, f, g, h)
    xhist = [x]

    # For theoritical reasons, tol should be smaller than (3-sqrt(5))/2
    if tol < (3-np.sqrt(5))/2:
        while gap > tol:
            x, gap = dampedNewtonStep(x, f, g, h)
            xhist.append(x)
        xstar = x
    else:
        raise ValueError("Enter a value for tol < (3-sqrt(5))/2")
        
    return xstar, xhist