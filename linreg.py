import numpy as np
from tabulate import tabulate

 
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
    


