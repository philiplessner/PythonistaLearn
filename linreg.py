import numpy as np
import matplotlib.pyplot as plt


def error(xi, yi, h_theta):
    '''
    Difference between predicted and observed value for a training example
    Parameters
        xi: x vector (length j+1) for training example i
        yi: y observation for training example i
        h_theta: vector of parameters (theta0...thetaj)
    Returns
        error (predicted - observed)
    '''
    return np.dot(xi, h_theta) - yi


def errors(X, y, h_theta):
    return np.dot(X, h_theta) - y
   

def J(X, y, h_theta):
    '''
    Cost function for multiple linear regression
    Parameters
        X: matrix of independent variables (i rows of observations and j cols
           of variables). x0=1 for all i
        y: dependent variable (i rows)
        h_theta: coefficients (j cols)
    Returns
        Cost function (sum of squares of errors)
    '''
    return sum(errors(X, y, h_theta)**2) / (2. * len(y))


def gradJ(X, y, h_theta):
    '''
    Gradient of Cost function for batch gradient descent for
    Multiple linear regression
    Parameters
        X: matrix of independent variables (i rows of observations and j cols
           of variables). x0=1 for all i
        y: dependent variable (i rows)
        h_theta: coefficients (j cols)
    Returns
        Gradient of cost function (j cols, one for each h_thetaj)
        Will be used to update h_theta i gradient descent
    '''
    return np.dot(X.T, errors(X, y, h_theta)) / len(y)


def JS(xi, yi, h_theta):
    return 0.5 * error(xi, yi, h_theta)**2


def gradJS(xi, yi, h_theta):
    '''
    Gradient of Cost function for stochastic gradient descent for
    Multiple linear regression
    Uses a single observation to compute gradient
    Parameters
        xi: x vector (length j+1) for training example i
        yi: y observation for training example i
        h_theta: vector of parameters (theta0...thetaj)
    Returns
        Gradient of cost function (j cols, one for each h_thetaj)
        Will be used to update h_theta in gradient descent
    '''
    return np.dot(xi, error(xi, yi, h_theta))
    

def plot_cost(cost):
    plt.semilogy(range(0, len(cost)), cost, 'b+')
    plt.show()
    plt.clf()

