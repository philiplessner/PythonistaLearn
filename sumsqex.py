import numpy as np
from toolz import take, compose, pluck
#import matplotlib.pyplot as plt
#from utility import until_within_tol
from fgd import gradient_descent
#from out_utils import plot_lrates


def f(x):
    return np.sum(x * x)
        

def df(x):
    return 2 * x


x0 = np.array([6., 33., 12.2])
tol = 1.e-6
al = [1., 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
niter = 100

#plot_lrates(f, df, x0, al, niter)

result = list(take(50, ((f(e), e) for e in gradient_descent(df, x0)) ))
#xs = ['x' + unicode(i) for i in xrange(len(x0))]
#table = pylsytable2(['y'] + xs)
#table.add_data('y', list(pluck(0, result)), '{:.2e}')
#for i, x in enumerate(xs):
#    table.add_data(x, list(pluck(i,pluck(1, result))), '{:.2e}')
#print(table)
print(result)
