import numpy as np
import matplotlib.pyplot as plt
from getmnist import load_mnist 
import NNplay as nn
from ml_util import encode_labels
from metrics import MScores



# Get the data
X_train, y_train = load_mnist('./data/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('./data/', kind='t10k')
print('Rows: %d, columns: %d'% (X_test.shape[0], X_test.shape[1]))

y_trainh = encode_labels(y_train, 10)

Z_train = X_train[0:1000].astype(float)
q_trainh = y_trainh[0:1000]
Q_train = Z_train / (255. * 0.99) + 0.01

# Set up the network
hyperparam = {'nhidden': 50,
              'eta': 0.01,
              'epochs': 500,
              'minibatches': 3}
print('****Training****')
train_paramf, cost = nn.fit(Q_train, q_trainh, hyperparam)

# Print out the results
prediction = nn.predict_proba(Q_train, train_paramf)
classes = nn.predict(Q_train, train_paramf)
print('\nsyn0\n', train_paramf[0])
print('\nb0\n', train_paramf[1])
print('\nsyn1\n', train_paramf[2])
print('\nb1\n', train_paramf[3])
print('\nClass Probabilities\n', prediction)
print('\nClass\n', classes)
print('\nError\n', cost[-1])

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(range(0, len(cost)), cost)
plt.show()
plt.clf()

score = MScores(q_trainh, encode_labels(np.array(classes), 10))
print('Precision: ', score.precision())
print('Recall: ', score.recall())
