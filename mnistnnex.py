import numpy as np
import matplotlib.pyplot as plt
from getmnist import load_mnist 
import NNplay as nn
from utility import Scaler


def encode_labels(y, noutputs):
    onehot = np.zeros((y.shape[0], noutputs)) + 0.01     
    for irow, val in enumerate(y):
         onehot[irow, val] = 0.99
    return onehot

# Get the data
X_train, y_train = load_mnist('./data/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('./data/', kind='t10k')
print('Rows: %d, columns: %d'% (X_test.shape[0], X_test.shape[1]))

y_trainh = encode_labels(y_train, 10)

Z_train = X_train[0:1000].astype(float)
q_trainh = y_trainh[0:1000]
Q_train = Z_train / (255. * 0.99) + 0.01

print(q_trainh) 


# Set up the network
hyperparam = {'nhidden': 50,
              'alpha': 0.3,
              'niter': 1000}

train_params = nn.fit(Q_train, q_trainh, hyperparam)

# Print out the results
train_paramf = train_params[-1]
prediction = nn.predict_proba(Q_train, train_paramf)
predictions = [nn.predict_proba(Q_train, tp) for tp in train_params]
errors = [nn.error(q_trainh, p) for p in predictions]
classes = nn.predict(Q_train, train_paramf)
print('\nsyn0\n', train_paramf[0])
print('\nb0\n', train_paramf[1])
print('\nsyn1\n', train_paramf[2])
print('\nb1\n', train_paramf[3])
print('\nClass Probabilities\n', prediction)
print('\nClass\n', classes)
print('\nError\n', errors[-1])

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(range(0, len(errors)), errors)
plt.show()
plt.clf()
