import numpy as np
import matplotlib.pyplot as plt
from toolz import compose


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(Z):
    return np.divide(np.exp(Z),
                     np.sum(np.exp(Z), axis=1).reshape((Z.shape[0], 1)))


def cross_entropy(Z, y_encoded):
    P = np.log(softmax(Z))
    return -1. * np.array([np.dot(y_encoded[i], P[i])
                           for i in range(y_encoded.shape[0])])


def SMC(X, y_encoded, W):
    Z = np.dot(X, W)
    return np.sum(cross_entropy(Z, y_encoded)) * 1. / y_encoded.shape[0]


def gradSMC(X, y_encoded, W):
    P = np.log(softmax(np.dot(X, W)))
    return np.dot(X.T,
                  np.subtract(y_encoded, P)) * -1. / y_encoded.shape[0]


def LLL(X, y, h_theta):
    return np.sum(y * np.log(logistic(np.dot(X, h_theta))) +
                  (1 - y) * np.log(1. - logistic(np.dot(X, h_theta))))


def errors(X, y, h_theta):
    return logistic(np.dot(X, h_theta)) - y


def gradL(X, y, h_theta):
    return np.dot(X.T, errors(X, y, h_theta))


def classify(probs):
    if probs.ndim == 1:
        return list(map(compose(int, round), probs))
    return np.argmax(probs, axis=1)


def plot_cost(cost):
    plt.plot(list(range(0, len(cost))), cost, 'b+')
    plt.show()
    plt.clf()
