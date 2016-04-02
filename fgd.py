import random
from functools import partial
from itertools import chain
from toolz import iterate, accumulate, curry


@curry
def gradient_step(df, eta, theta):
    '''
    Calculate theta_k+1 from theta_k
    by taking step in negative direction of gradient
    theta is a j dimensional vector
    Parameters
        df: Gradient of function f [df1, df2,...,dfj]
        eta: Learning rate
        theta_k: [theta_k1, theta_k2,...,theta_kj]
    Returns
       [theta_k+11, theta_k+12,...,thetak_k+1j]
    '''
    return theta - eta * df(theta)
    

def gradient_descent(df, theta_0, eta=0.1):
    '''
    Parameters
        df: Gradient of function f
        theta0: Initial guess, theta ia a j dimensional vector ([theta_01,
        theta_02,...,theta0_0j])
        eta: Learning rate
    Returns
        Generator sequence of [theta_k1, theta_k2,...,theta_kj]
        where k = 0 to ...
    '''
    return iterate(gradient_step(df, eta), theta_0)


def sgd_step(df, eta, theta_k, xy_i):
    '''
    df is a function of x_i, y_i, theta
    '''
    x_i, y_i = xy_i
    gradient = df(x_i, y_i, theta_k)
    return [theta_k - eta * df_k
            for theta_k, df_k in zip(theta_k, gradient)]


def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
    random.shuffle(indexes)                    # shuffle them
    for i in indexes:                          # return the data in that order
        yield data[i]


def sgd(df, X, y, theta_0, eta=0.1):
    '''
    Parameters
        df: Gradient of function f
        X: Matrix of features
        y: vector of observations
        theta0: Initial guess, theta ia a j dimensional vector ([theta_01,
        theta_02,...,theta0_0j])
        eta: Learning rate
    Returns
        Generator sequence of [theta_k1, theta_k2,...,theta_kj] 
        where k = 0 to ...
    '''
    xys = chain([theta_0], in_random_order(list(zip(X, y))))
    return accumulate(partial(sgd_step, df, eta), xys)

