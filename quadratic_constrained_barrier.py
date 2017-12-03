#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""  This script implements the functions necessary for solving the following
optimization problem using the barrier method:

minimize(x) 1/2*xTQx + pTx 
st. Ax <= b

Rewritten as:

minimize(x) t(1/2*xTQx + pTx) + B(b - Ax) 
where B is a self-concordant barrier for the set Ax <= b
We use the logarithmic barrier B(x) = -sum(log(xi))
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

import numpy as np 

from newton_method_line_search import newtonLS
from damped_newton_method import dampedNewton

def log_barrier(x):
    """ Compute the logarithmic barrier at x.

    Parameters
    ----------

    x: array

    Returns
    -------

    log_barrier: float
    """
    x = - np.sum(np.log(x))
    return x

def phi(x, t, Q, p, A, b):
    """ Compute the value of the function of x:
    phi = t(1/2*xTQx + pTx) + B(b - Ax) 

    where Q, p, A, b are the matrices of the quadratic problem:

    minimize(x) 1/2*xTQx + pTx 
    st. Ax <= b

    Parameters
    ----------

    x: array, shape = [d]
    t: float
    Q: array, shape = [d, d]
    p: array, shape = [d]
    A: array, shape = [n, d]
    b: array, shape = [n]

    Returns
    -------

    phi: float
    """
    x = t * (1/2*np.dot(x, Q.dot(x)) + p.dot(x)) + log_barrier(b-A.dot(x))
    return x

def grad(x, t, Q, p, A, b):
    """ Compute the gradient of the function of x:
    phi = t(1/2*xTQx + pTx) + B(b - Ax) 
    
    where Q, p, A, b are the matrices of the quadratic problem:

    minimize(x) 1/2*xTQx + pTx 
    st. Ax <= b

    Parameters
    ----------

    x: array, shape = [d]
    t: float
    Q: array, shape = [d, d]
    p: array, shape = [d]
    A: array, shape = [n, d]
    b: array, shape = [n]

    Returns
    -------

    grad: array, shape = [d]
    """
    x = t*(Q.dot(x) + p) + np.sum(np.divide(A.T, b-A.dot(x)), axis = 1)
    
    return x

def hess(x, t, Q, p, A, b):
    """ Compute the hessian of the function of x:
    phi = t(1/2*xTQx + pTx) + B(b - Ax) 
    
    where Q, p, A, b are the matrices of the quadratic problem:

    minimize(x) 1/2*xTQx + pTx 
    st. Ax <= b

    Parameters
    ----------

    x: array, shape = [d]
    t: float
    Q: array, shape = [d, d]
    p: array, shape = [d]
    A: array, shape = [n, d]
    b: array, shape = [n]

    Returns
    -------

    hess: array, shape = [d, d]
    """
    # Divide each column i of A is divided by bi- (Ax)i
    A_ = np.divide(A.T, b-A.dot(x))

    x = t*Q + A_.dot(A_.T)

    return x

def barr_method(Q, p, A, b, x_0, t_0, mu, tol, method = "dampedNewton", 
                verbose = False):
    """Barrier method for solving a quadratic problem with inequality 
    constraints.

    Parameters
    ----------
    
    Q: array
    p: array
    A: array
    b: array
    x_0: array
        Strictly feasible starting point.
    t_0: float
        Initial t.
    mu: int
        Factor of increase of t in the outer iteration.
    tol: float
        Tolerance of the barrier method.
    method: str (optional)
        Optimization method used in the centering step.
        1. Newton with backtracking line-search ("newtonLS")
        2. Damped Newton ("dampedNewton")

    Returns
    -------
    x_sol: array
        Optimal solution.
    xhist: list (array)
        History of x.
    outer_iterations: list
        The i-th element is the number of inner iterations for the i-th outer
        iteration.


    """
    # Store the number of inner iterations per outer iteration (the i-th 
    # element of the list corresponds to the number of inner iterations for the 
    # i-th outer iteration)
    outer_iterations = []

    # Number of inequality constraints
    m = b.shape[0]

    # Check if x_0 is strictly feasible
    if np.sum(A.dot(x_0) < b) == m:

        t = t_0
        x = x_0

        # History of all Newton updates (+ the initial point)
        xhist = [x_0]

        while m/t >= tol:
            # Update phi depending on t
            f = lambda x: phi(x, t, Q, p, A, b)
            g = lambda x: grad (x, t, Q, p, A, b)
            h = lambda x: hess (x, t, Q, p, A, b)

            if method == "newtonLS":
                x, xhist_Newton = newtonLS(x, f, g, h, tol, A, b)
            elif method == "dampedNewton":
                x, xhist_Newton = dampedNewton(x, f, g, h, tol)
            else:
                raise ValueError("Enter a correct method: 'newtonLS' or 'dampedNewton'")

            xhist += xhist_Newton
            outer_iterations += [len(xhist_Newton)]

            t *= mu      

            if verbose:
                print("Outer iteration number {} completed".format(len(outer_iterations)))

        x_sol = x

    else:
        raise ValueError("x_0 is not scritly feasible, cannot proceed")

    return x_sol, xhist, outer_iterations 

def test_log_barrier():
    """ Test the implementation of the log barrier.
    """
    test = log_barrier(np.array([np.e, np.e**2]))
    assert test == -3

def test_grad():
    """ Test the implementation of the gradient of phi.
    """
    p = np.array([1, 2])
    Q = np.identity(2)
    b = np.array([1, 2])
    A = np.array([[1, 2], [3, 4]])
    t = 1
    x = np.array([0, 0])
    test = grad(x, t, Q, p, A, b)
    print(test)
    assert np.amax(np.fabs(test - np.array([3.5, 6]))) <= 1e-6

def test_hess():
    """ Test the implementation of the hessian of phi.
    """
    p = np.array([1, 2])
    Q = np.identity(2)
    b = np.array([1, 2])
    A = np.array([[1, 2], [3, 4]])
    t = 1
    x = np.array([0, 0])
    test = hess(x, t, Q, p, A, b)
    print(test)
    assert np.amax(np.fabs(test - np.array([[4.25, 5], [5, 9]]))) <= 1e-6

# Perform tests
if __name__ == '__main__':
    test_log_barrier()
    test_grad()
    test_hess()