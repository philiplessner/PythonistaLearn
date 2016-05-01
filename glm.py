from functools import partial
import numpy as np
from toolz import curry
import fgd
from utility import prepend_x0


@curry
def fit(cost_f, cost_df, hyperparam, data):
    '''
    Compute values of multiple linear regression coefficients
    Parameters
        cost_f: Cost function
        cost_df: gradient of cost function
        data: list of tuples [(Xi, yi)]
        X: matrix of independent variables (i rows of observations and j cols
           of variables).
        y: dependent variable (i rows)
        hyperparameters:
            eta: learning rate
            epochs: number of passing over the training set
            minibatches: number of minbatches
            adaptive: learning rate decrease
    Returns
        Fitting parameters (j cols)
    '''
    Q, w = list(zip(*data))
    X = prepend_x0(np.array(Q))
    y = np.array(w)
    eta = hyperparam['eta']
    epochs = hyperparam['epochs']
    minibatches = hyperparam['minibatches']
    adaptive = hyperparam['adaptive']
    indexes = np.array_split(range(y.shape[0]), minibatches)
    h_theta = np.random.uniform(0.0, 1.0, (X.shape[1], y.shape[1]))
    eta_new = eta
    cost = [cost_f(X, y, h_theta)]
    for i in range(epochs):
        np.random.seed(i)
        for index in np.random.permutation(indexes):
            df = partial(cost_df, X[index], y[index])
            h_theta = fgd.gradient_step(df, eta_new, h_theta)
        cost.append(cost_f(X, y, h_theta))
        eta_new = adaptive * eta_new
    return h_theta, cost


@curry
def predict(f, X, h_theta):
    return f(np.dot(prepend_x0(X), h_theta))
