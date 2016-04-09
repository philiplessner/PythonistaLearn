from functools import partial
import numpy as np
from toolz import curry
import fgd


@curry
def fit(cost_f, cost_df, hyperparam, h_theta0, data):
    '''
    Compute values of multiple linear regression coefficients
    Parameters
        cost_f: Cost function
        cost_df: gradient of cost function
        h_theta0: initial guess for fitting parameters (j cols)
        data: list of tuples [(Xi, yi)]
        X: matrix of independent variables (i rows of observations and j cols
           of variables). x0=1 for all i
        y: dependent variable (i rows)
        eta: learning rate
        it_max: maximum number of iterations
    Returns
        Fitting parameters (j cols)
    '''
    Q, w = list(zip(*data))
    X = np.array(Q)
    y = np.array(w)
    eta = hyperparam['eta']
    epochs = hyperparam['epochs']
    minibatches = hyperparam['minibatches']
    adaptive = hyperparam['adaptive']
    indexes = np.array_split(range(y.shape[0]), minibatches)
    h_theta = h_theta0
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
    return f(np.dot(X, h_theta))

