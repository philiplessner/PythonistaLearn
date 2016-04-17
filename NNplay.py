import numpy as np


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))
    

def dlogistic(x):
    return x * (1.0 - x)


def initialize(nexamples, nfeatures, nhidden, noutputs):
    np.random.seed(1)
    syn0 = 2*np.random.random((nfeatures, nhidden)) - 1
    b0 = 2. * np.random.random((1, nhidden)) - 1
    syn1 = 2*np.random.random((nhidden, noutputs)) - 1
    b1 = 2. * np.random.random((1, noutputs)) - 1
    return syn0, b0, syn1, b1


def backprop(X, y, eta, train_param):
    # Foward Propagation
    syn0, b0, syn1, b1 = train_param
    l1 = logistic(np.dot(X, syn0) + b0)
    l2 = logistic(np.dot(l1, syn1) + b1)
    # Back Propagation
    l2_error = y - l2
    l2_delta = l2_error * dlogistic(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * dlogistic(l1)
    # Gradient Descent
    return (syn0 + eta * (X.T.dot(l1_delta)),
            b0 + eta * np.sum(l1_delta, axis=0, keepdims=True),
            syn1 + eta * (l1.T.dot(l2_delta)),
            b1 + eta * np.sum(l2_delta, axis=0, keepdims=True))


def fit(X, y, hyperparam):
    eta = hyperparam['eta']
    epochs = hyperparam['epochs']
    nhidden = hyperparam['nhidden']
    minibatches = hyperparam['minibatches']
    nfeatures = X.shape[1]
    nexamples = X.shape[0]
    noutputs = y.shape[1]
    train_param0 = initialize(nexamples, nfeatures, nhidden, noutputs)
    indexes = np.array_split(range(y.shape[0]), minibatches)
    train_param = train_param0
    cost = [error(y, predict_proba(X, train_param))]
    for i in range(epochs):
        np.random.seed(i)
        for index in np.random.permutation(indexes):
            train_param = backprop(X[index], y[index], eta, train_param)
        cost.append(error(y, predict_proba(X, train_param)))
    return train_param, cost


def predict_proba(X, train_param):
    syn0, b0, syn1, b1 = train_param
    l1 = logistic(np.dot(X, syn0) + b0)
    return logistic(np.dot(l1, syn1) + b1)


def predict(X, train_param):
    return np.argmax(predict_proba(X, train_param), axis=1)
    
    
def error(y, prediction):
    return (0.5 / y.shape[0]) * np.sum((prediction - y)**2)

