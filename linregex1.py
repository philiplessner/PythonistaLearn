import numpy as np
from toolz import compose, identity, pluck
import linreg as lr
import glm
import metrics
from utility import Scaler, prepend_x0
from ml_util import train_test_split
from out_utils import plot_cost, plot_errors


# Get the data
input = np.loadtxt('./data/Folds.csv', delimiter=',', skiprows=1)
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


print('****Minibatch Gradient Descent****')
print('\n--Training--\n')
hyperparam = {'eta': 0.3,
              'epochs': 300,
              'minibatches': 1,
              'adaptive': 0.99}
print('\nHyperparamters\n')
for k, v in hyperparam.items():
    print(k, '\t', v)
print('\nNumber of Training Examples: ', X_train.shape[0], '\n')

h_thetaf, cost = glm.fit(lr.J,
                        lr.gradJ,
                        hyperparam, 
                        h_theta0)(scaledtrain_data)
plot_cost(cost)
h_thetad = scale.denormalize(h_thetaf)
print('Coefficients\t', h_thetaf)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + str(i), '\t', h_theta)
yp_train = glm.predict(identity, X_train, h_thetaf)
plot_errors(y_train, yp_train)
corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('\n--Testing--')
yp_test = glm.predict(identity, X_test, h_thetaf)
plot_errors(y_test, yp_test)
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

