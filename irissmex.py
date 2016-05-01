import random
import numpy as np
import matplotlib.pyplot as plt
import glm
import logreg as lr
from utility import csv_reader, Scaler, prepend_x0
from ml_util import train_test_split
from out_utils import logistic_table
from metrics import MScores


def softmax_classes(prediction):
    return np.argmax(prediction, axis=1)


def encode_labels(y, noutputs):
    onehot = np.zeros((y.shape[0], noutputs), dtype=np.float64)
    for irow, val in enumerate(y):
        onehot[irow, val.astype(np.uint8)] = 1.0
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
scaledX_train = prepend_x0(scale.transform(Z_train))
Z_test, y_test = zip(*test_data)
y_test = np.array(y_test)
scaledX_test = prepend_x0(scale.transform(Z_test))

weights0 = np.random.uniform(0.0, 1.0,
                             (scaledX_train.shape[1], y_train.shape[1]))
hyperparam = {'eta': 0.5,
              'epochs': 1000,
              'minibatches': 4,
              'adaptive': 0.98}

weightsf, cost = glm.fit(lr.SMC,
                         lr.gradSMC,
                         hyperparam,
                         weights0,
                         zip(scaledX_train, y_train))

# Print out the results
print('****Training****\n')
prediction = glm.predict(lr.softmax, scaledX_train, weightsf)
classes = softmax_classes(prediction)
print('\nWeight Matrix\n', weightsf)
print('\nError\n', cost[-1])

logistic_table(prediction, classes, np.argmax(np.array(y_train), axis=1))
score = MScores(y_train, encode_labels(np.array(classes), 3))
print('Precision: ', score.precision())
print('Recall: ', score.recall())
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(range(0, len(cost)), cost)
plt.show()
plt.clf()

print('****Testing****\n')
prediction = glm.predict(lr.softmax, scaledX_test, weightsf)
classes = softmax_classes(prediction)
logistic_table(prediction, classes, np.argmax(np.array(y_test), axis=1))
score = MScores(y_test, encode_labels(np.array(classes), 3))
print('Precision: ', score.precision())
print('Recall: ', score.recall())
