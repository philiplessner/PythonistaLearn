from tabulate import tabulate


def logistic_table(probs, yp, y):   
    print(tabulate(list(zip(probs, yp, y)), 
                headers=['Probability', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))
