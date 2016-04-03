import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


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
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.semilogy(range(0, len(cost)), cost, '+')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    plt.show()
    plt.clf()
    

def plot_errors(y, yp):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(0, len(y)), (y - yp) / y * 100.)
    ax.set_xlabel('Point')
    ax.set_ylabel('% Error')
    plt.show()
    plt.clf()


def print_predictions(y, yp):
    print(tabulate(list(zip(yp, y)),
                        headers=['yp', 'yi'],
                        tablefmt='fancy_grid')) 
