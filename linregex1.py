from functools import partial
from toolz import compose, identity, pluck
from tabulate import tabulate
from utility import csv_reader, Scaler, prepend_x0
import metrics
import linreg as lr
import glm
from ml_util import train_test_split
import numpy as np
from numpy.linalg import lstsq

# Get the data
input = np.loadtxt('./data/Folds_medium.csv', delimiter=',', skiprows=1)
Z = np.array(list(pluck(list(range(0, len(input[0])-1)), input)))
y = np.array(list(pluck(len(input[0])-1, input)))
data = zip(Z, y)
# Split into a train set and test set
train_data, test_data = train_test_split(data, 0.33)
# Scale the training data
scale = Scaler()
Z_train, y_train = zip(*train_data)
scale.fit(Z_train)
transform = compose(prepend_x0, scale.transform)
X_train = transform(Z_train)
scaledtrain_data = list(zip(X_train, y_train))
# Scale the testing data using the same scaling parameters
# used for the training data
Z_test, y_test = zip(*test_data)
X_test = transform(Z_test)

h_theta0 = np.array([0., 0., 0., 0., 0.])
print('****Gradient Descent****')
h_thetaf, cost = glm.fit(lr.J, 
                        lr.gradJ, 
                        h_theta0, 
                        eta=0.3, 
                        it_max=300, gf='gd')(scaledtrain_data)
lr.plot_cost(cost)
h_thetad = scale.denormalize(h_thetaf)
yp_train = glm.predict(identity, X_train, h_thetaf)

print('\n--Training--')
print('Coefficients\t', h_thetaf)
print(tabulate(list(zip(yp_train, y_train)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
    

print('Coefficients\t', h_thetad)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + str(i), '\t', h_theta)

corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('\n--Testing--')
yp_test = glm.predict(identity, X_test, h_thetaf)
print(tabulate(list(zip(yp_test, y_test)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

print('\n\n****Stochastic Gradient Descent****')
print('\n--Training--')
h_thetaf, cost = glm.fit(lr.JS,
                         lr.gradJS,
                         h_theta0, 
                         eta=0.1,
                         it_max=300, 
                         gf='sgd')(scaledtrain_data)

lr.plot_cost(cost)
print('Coefficients\t', h_thetaf)
yp_train = glm.predict(identity, X_train, h_thetaf)
h_thetad = scale.denormalize(h_thetaf)
print(tabulate(list(zip(yp_train, y_train)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
print('Coefficients\t', h_thetad)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + str(i), '\t', h_theta)

corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('\n--Testing--')
yp_test = glm.predict(identity, X_test, h_thetaf)
print(tabulate(list(zip(yp_test, y_test)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

print('****Minibatch Gradient Descent****')
hyperparam = {'eta': 0.3,
              'epochs': 300,
              'minibatches': 10}
h_thetaf, cost = glm.fit2(lr.J, 
                        lr.gradJ,
                        hyperparam, 
                        h_theta0)(scaledtrain_data)
lr.plot_cost(cost)
h_thetad = scale.denormalize(h_thetaf)
yp_train = glm.predict(identity, X_train, h_thetaf)

print('\n--Training--')
print('Coefficients\t', h_thetaf)
print(tabulate(list(zip(yp_train, y_train)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
    

print('Coefficients\t', h_thetad)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + str(i), '\t', h_theta)

corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('\n--Testing--')
yp_test = glm.predict(identity, X_test, h_thetaf)
print(tabulate(list(zip(yp_test, y_test)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

print('****Numpy Solution****')
Q = np.array([e + [1.] for e in Z])
coeff = lstsq(Q, np.array(y))
print(coeff)

