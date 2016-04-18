import random
import numpy as np
import matplotlib.pyplot as plt
import NNplay as nn
from utility import csv_reader, Scaler
from ml_util import train_test_split
from out_utils import logistic_table, plot_decision_boundary
from metrics import MScores


def encode_labels(y, noutputs):
    onehot = np.zeros((y.shape[0], noutputs)) 
    for irow, val in enumerate(y):
         onehot[irow, val] = 1.0
    return onehot


# Get the iris data set
# SL: sepal length, SW: Sepal Width, PL: Petal Length, PW: Petal Width
# 0:  Iris Setosa 1: Iris Versicolour 2: Iris Virginica
Z, q = csv_reader('./data/iris.csv', ['SL', 'SW', 'PL', 'PW'], 'Type')

t = list(zip(Z, q))
random.shuffle(t)
W, u = list(zip(*t))
yencoded = encode_labels(np.array(u), 3)
train_data, test_data = train_test_split(zip(W, yencoded), 0.33)
# Scale data
Z_train, y_train = zip(*train_data)
y_train = np.array(y_train) 
scale = Scaler()
scale.fit(Z_train)
scaledX_train = scale.transform(Z_train)
Z_test, y_test = zip(*test_data)
y_test = np.array(y_test)
scaledX_test = scale.transform(Z_test)

# Train the Network
hyperparam = {'nhidden': 2,
              'eta': 0.03,
              'epochs': 5000,
              'minibatches': 3}    
train_paramf, cost = nn.fit(scaledX_train, y_train, hyperparam)

# Print out the results
print('****Training****\n')
prediction = nn.predict_proba(scaledX_train, train_paramf)
classes = nn.predict(scaledX_train, train_paramf)
print('\nsyn0\n', train_paramf[0])
print('\nb0\n', train_paramf[1])
print('\nsyn1\n', train_paramf[2])
print('\nb1\n', train_paramf[3])
print('\nError\n', cost[-1])

logistic_table(prediction, classes, np.argmax(np.array(y_train), axis=1))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(range(0, len(cost)), cost)
plt.show()
plt.clf()

print('****Testing****\n')
prediction = nn.predict_proba(scaledX_test, train_paramf)
classes = nn.predict(scaledX_test, train_paramf)
logistic_table(prediction, classes, np.argmax(np.array(y_test), axis=1))
score = MScores(y_test, encode_labels(np.array(classes), 3))
print('Precision: ', score.precision())
print('Recall: ', score.recall())

# plot_decision_boundary(nn.predict, np.array(Z), train_paramf)

