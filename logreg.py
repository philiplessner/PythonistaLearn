import numpy as np
import matplotlib.pyplot as plt
from toolz import compose


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

    
def LLL(X, y, h_theta):
    return np.sum(y * np.log(logistic(np.dot(X, h_theta))) + (1 - y) * np.log(1. - logistic(np.dot(X, h_theta))))


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
