from functools import partial
import numpy as np
from toolz import take, curry, pluck
import fgd


@curry
def fit(cost_f, cost_df, h_theta0, data, eta=0.1, it_max=500, gf='gd'):
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
    if gf == 'gd':
        f = partial(cost_f, X, y)
        df = partial(cost_df, X, y)
        ans = list(take(it_max,
                        ((h_theta, f(h_theta)) for h_theta in 
                          fgd.gradient_descent(df, h_theta0, eta=eta))))
        value = np.array(list(pluck(0, ans)))
        cost = np.array(list(pluck(1, ans)))
        return value[-1], cost
    elif gf == 'sgd':
        df = cost_df
        cost = [sum(cost_f(xi, yi, h_theta0) for xi, yi in data)]
        h_theta = h_theta0
        eta_new = eta
        for _ in range(it_max):
            ans = list(take(len(y), (e for e in fgd.sgd(df, X, y, h_theta, eta=eta_new))))
            h_theta = np.array(ans[-1])
            cost.append(sum(cost_f(xi, yi, h_theta) for xi, yi in data))
            eta_new = 0.99 * eta_new
        return h_theta, cost
    else:
        print('Not a valid function')
        return


@curry
def fit2(cost_f, cost_df, hyperparam, h_theta0, data):
    Q, w = list(zip(*data))
    X = np.array(Q)
    y = np.array(w)
    eta = hyperparam['eta']
    epochs = hyperparam['epochs']
    minibatches = hyperparam['minibatches']
    indexes = np.array_split(range(y.shape[0]), minibatches)
    print(indexes)
    h_theta = h_theta0
    eta_new = eta
    cost = [cost_f(X, y, h_theta)]
    for i in range(epochs):
        np.random.seed(i) 
        for index in np.random.permutation(indexes):
            df = partial(cost_df, X[index], y[index])
            h_theta = fgd.gradient_step(df, eta_new, h_theta)
        cost.append(cost_f(X, y, h_theta))
        eta_new = 0.99 * eta_new
    return h_theta, cost
    

@curry
def predict(f, X, h_theta):
    return f(np.dot(X, h_theta))
