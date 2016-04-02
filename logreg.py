import numpy as np
import matplotlib.pyplot as plt
from toolz import compose


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_log_likelihood_i(x_i, y_i, h_theta):
    return y_i * np.log(logistic(np.dot(x_i, h_theta))) + (1 - y_i) * np.log(1 - logistic(np.dot(x_i, h_theta)))
       
    
def LLL(X, y, h_theta):
    return sum(y * np.log(logistic(np.dot(X, h_theta))) + (1 - y) * np.log(1. - logistic(np.dot(X, h_theta))))


def logistic_log_partial_ij(x_i, y_i, h_theta, j):
    """here i is the index of the data point,
    j the index of the derivative"""
    return (logistic(np.dot(x_i, h_theta)) - y_i) * x_i[j]
   
    
def logistic_log_gradient_i(x_i, y_i, h_theta):
    """the gradient of the log likelihood
    corresponding to the ith data point"""
    return [logistic_log_partial_ij(x_i, y_i, h_theta, j) for j, _ in enumerate(h_theta)]
    

def errors(X, y, h_theta):
    return logistic(np.dot(X, h_theta)) - y
    

def gradL(X, y, h_theta):
    return np.dot(X.T, errors(X, y, h_theta))


def logistic_classes(logistic_probs):
    return list(map(compose(int, round), logistic_probs))
  

def plot_cost(cost):
    plt.plot(list(range(0, len(cost))), cost, 'b+')
    plt.show()
    plt.clf()
