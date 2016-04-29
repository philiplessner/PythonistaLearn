import numpy as np
from utility import prepend_x0


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def dlogistic(x):
    return x * (1.0 - x)


def initialize(nexamples, nfeatures, nhidden, noutputs):
    np.random.seed(1)
    syn0 = 2*np.random.random((nfeatures, nhidden)) - 1
    syn1 = 2*np.random.random((nhidden + 1, noutputs)) - 1
    return syn0, syn1


def backprop(X, y, eta, train_param):
    # Foward Propagation
    syn0, syn1 = train_param
    l1 = prepend_x0(logistic(np.dot(X, syn0)))
    l2 = logistic(np.dot(l1, syn1))
    # Back Propagation
    l2_error = y - l2
    l2_delta = l2_error * dlogistic(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * dlogistic(l1)
    l1_delta = l1_delta[:, 1:]
    # Gradient Descent
    return (syn0 + eta * (X.T.dot(l1_delta)),
            syn1 + eta * (l1.T.dot(l2_delta)))


def fit(Z, y, hyperparam):
    eta = hyperparam['eta']
    decrease_rate = hyperparam['decrease_rate']
    epochs = hyperparam['epochs']
    nhidden = hyperparam['nhidden']
    minibatches = hyperparam['minibatches']
    X = prepend_x0(Z)
    nfeatures = X.shape[1]
    nexamples = X.shape[0]
    noutputs = y.shape[1]
    train_param0 = initialize(nexamples, nfeatures, nhidden, noutputs)
    indexes = np.array_split(range(y.shape[0]), minibatches)
    train_param = train_param0
    cost = [error(y, predict_proba(Z, train_param))]
    for i in range(epochs):
        eta = eta / (1. + i * decrease_rate)
        np.random.seed(i)
        for index in np.random.permutation(indexes):
            train_param = backprop(X[index], y[index], eta, train_param)
        cost.append(error(y, predict_proba(Z, train_param)))
    return train_param, cost


def predict_proba(Z, train_param):
    syn0, syn1 = train_param
    X = prepend_x0(Z)
    l1 = prepend_x0(logistic(np.dot(X, syn0)))
    return logistic(np.dot(l1, syn1))


def predict(Z, train_param):
    return np.argmax(predict_proba(Z, train_param), axis=1)


def error(y, prediction):
    term1 = -y * np.log(prediction)
    term2 = (1 - y) * np.log(1 - prediction)
    return np.sum(term1 - term2)
