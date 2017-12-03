#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" This class implements a linear SVM (primal and dual) with two optimization
methods: the damped Newton algorithm and the backtracking line-search one.
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

import numpy as np
import time

from coordinate_descent import coordinate_descent_svm_dual
from quadratic_constrained_barrier import barr_method
from transform_svm import transform_svm_primal
from transform_svm import transform_svm_dual
from tools import give_available_values

class SVM():

    def __init__(self, tau = 1, t_0 = 1, tol = 0.0001, mu = 15, 
                 optimization = "primal", optimization_method = "barrier",
                 barrier_centering_method = "dampedNewton",
                 verbose = False):
        """ Train the SVM on the observation X using the labels y.

        Parameters
        ----------
        
        tau: float
            Regularization parameter (soft margin SVM).
        tol: float
            Tolerance of the barrier method.
        mu: int
            Factor of increase of t in the outer iteration of the barrier
            method.
        optimization: str (optional)
            Choose which problem to solve: "primal" or "dual".
        optimization_method: str (optional)
            Optimization method used
            1. Barrier method ("barrier")
            2. Coordinate descent method ("coordinateDescent")
        barrier_centering_method: str (optional)
            Optimization method used in the centering step of the barrier
            method.
            1. Newton with backtracking line-search ("newtonLS")
            2. Damped Newton ("dampedNewton")
        verbose: bool (optional)
            True to print the time taken for the training.
        """

        # Values checks
        optimization_values = ["primal", "dual"]
        optimization_method_values = ["barrier", "coordinateDescent"]
        barrier_centering_method_values = ["newtonLS", "dampedNewton"]

        if optimization not in optimization_values:
            raise ValueError("Enter a correct name for the problem to optimize:"
                             " 'primal' or 'dual'")
        if optimization_method not in optimization_method_values:
            raise ValueError("Enter a correct optimization method:"
                              " {}".format(give_available_values(
                                         optimization_method_values)))
        if barrier_centering_method not in barrier_centering_method_values:
            raise ValueError("Enter a correct optimization method for the"
                             " centering step of the barrier method: {}"
                              .format(give_available_values(
                                      barrier_centering_method_values)))

        # The coordinate descent method does not a "sub-optimization"
        if optimization_method == "coordinateDescent":
            barrier_centering_method = None

        if optimization == "primal" and optimization_method == "coordinateDescent":
           raise ValueError("The coordinate descent method is only implemented"
                             " for solving the 'dual' problem")

        self.tau = tau
        self.t_0 = t_0
        self.tol = tol
        self.mu = mu
        self.optimization = optimization
        self.optimization_method = optimization_method
        self.barrier_centering_method = barrier_centering_method
        self.verbose = verbose

    def train(self, X, y):
        """ Train the SVM on the observation X using the labels y.

        Parameters
        ----------
        
        X: array, shape = [n, d]
            Training observations.
        y: array, shape = [n, 1]
            True labels.

        Returns
        ----------

        self: object
            Returns self.
        """
        start = time.time()

        # Number of training examples
        self.n = X.shape[0]

        # Dimension (number of features)
        self.d = X.shape[1]

        # Add offset to the data points and make X of shape (d+1, n)
        X = np.vstack((X.T, np.ones(self.n)))

        if self.optimization == "dual":

            # Strictly feasible point for the dual
            self.x_0 = (1/(100*self.tau*self.n))*np.ones(self.n)

            # Formulation as a quadratic problem
            self.Q, self.p, self.A, self.b = transform_svm_dual(self.tau, X, y)

            if self.optimization_method == "coordinateDescent":
                self.x_sol, self.xhist = coordinate_descent_svm_dual(X, y, 
                    self.tau, self.Q, self.p, self.x_0, self.tol)

            else:

                # Solve the quadratic problem using the barrier method
                self.x_sol, self.xhist, self.outer_iterations = barr_method(self.Q, 
                    self.p, self.A, self.b, self.x_0, self.t_0, self.mu, self.tol, 
                    method = self.barrier_centering_method)

            # Normal vector to the separating hyperplane, shape (d+1, )
            self.w = self.x_sol.dot((X*y).T)

        elif self.optimization == "primal":

            # Strictly feasible point for the primal
            self.x_0 = np.zeros(self.d+1+self.n)
            self.x_0[self.d + 1:] = 1.1

            # Formulation as a quadratic problem
            self.Q, self.p, self.A, self.b = transform_svm_primal(self.tau, X, y)

            # Solve the quadratic problem using the barrier method
            self.x_sol, self.xhist, self.outer_iterations = barr_method(self.Q, 
                self.p, self.A, self.b, self.x_0, self.t_0, self.mu, self.tol, 
                method = self.barrier_centering_method)

            # Normal vector to the separating hyperplane, shape (d+1, )
            self.w = self.x_sol[:self.d + 1]

        stop = time.time()

        # Time for training
        t = round(stop-start, 2)
        if self.verbose:
            print("Time taken for training: {}s".format(t))

    def predict(self, X_test, y_test):
        """ Predict the labels of test data with the trained SVM and compute
        the mean accuracy.

        Parameters
        ----------
        
        X_test: array, shape = [n_test, d]
            Test observations.
        y_test: array, shape = [n_test, 1]
            True labels of the test data.

        Returns
        -------
        
        y_pred: array, shape = [n_test, 1]
            Predicted labels.
        accuracy: float
            Mean accuracy.
        """
        # Number of training examples
        self.n_test = X_test.shape[0]

        # Add offset to the data points and make X of shape (d+1, n)
        X_test = np.vstack((X_test.T, np.ones(self.n_test)))

        # Predict
        y_pred = np.sign(self.w.T.dot(X_test))

        # Compute the mean accuracy on the predictions
        accuracy = self.compute_mean_accuracy(y_pred, y_test)

        return y_pred, accuracy

    def compute_mean_accuracy(self, y_pred, y_test):
        """ Compute the mean accuracy of the predictions given the test data.

        Parameters
        ----------
        
        y_pred: array, shape = [n_samples, 1]
            Predicted labels on n_samples observations.
        y_test: array, shape = [n_samples, 1]
            True labels.

        Returns
        -------

        accuracy: float
            Mean accuracy.
        """
        accuracy = np.sum(y_pred == y_test)
        accuracy /= np.shape(y_test)[0]
        return accuracy


