import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def logistic_table(probs, yp, y):   
    print(tabulate(list(zip(probs, yp, y)), 
                headers=['Probability', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))
                

def plot_cost(cost):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.semilogy(range(0, len(cost)), cost, '+')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    plt.show()
    plt.clf()
    

def plot_errors(y, yp):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(0, len(y)), (y - yp) / y * 100.)
    ax.set_xlabel('Point')
    ax.set_ylabel('% Error')
    plt.show()
    plt.clf()


def print_predictions(y, yp):
    print(tabulate(list(zip(yp, y)),
                        headers=['yp', 'yi'],
                        tablefmt='fancy_grid'))


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func, X, fit_param):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()], fit_param)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

