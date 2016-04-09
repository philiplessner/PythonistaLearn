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

