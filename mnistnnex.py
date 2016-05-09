import time
import numpy as np
import matplotlib.pyplot as plt
from getmnist import load_mnist
import NNplay as nn
from ml_util import encode_labels
from metrics import MScores


def plot_misclass(X_test, y_test, y_test_pred):
    miscl_img = X_test[y_test[:y_test_pred.shape[0]] != y_test_pred][:25]
    correct_lab = y_test[y_test[:y_test_pred.shape[0]] != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test[:y_test_pred.shape[0]] != y_test_pred][:25]
    fig, ax = plt.subplots(nrows=5,
                           ncols=5,
                           sharex=True,
                           sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img,
                     cmap='Greys',
                     interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d'
                        % (i + 1, correct_lab[i], miscl_lab[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


# Get the data
X_train, y_train = load_mnist('./data/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('./data/', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

y_trainh = encode_labels(y_train, 10)

Z_train = X_train[0:6000].astype(float)
q_trainh = y_trainh[0:6000]
Q_train = Z_train / (255. * 0.99) + 0.01
Q_test = X_test[0:2000].astype(float) / (255 * 0.99) + 0.01
y_testh = encode_labels(y_test, 10)
q_testh = y_testh[0:2000]

# Set up the network
hyperparam = {'nhidden': 100,
              'eta': 0.01,
              'decrease_rate': 0.00001,
              'epochs': 500,
              'minibatches': 20}
print('****Training****')
print('\n--Hyperameters--')
for k, v in hyperparam.items():
    print(k, ':', v)
start_train = time.time()
train_paramf, cost = nn.fit(Q_train, q_trainh, hyperparam)
print('\nTraining Time:', time.time() - start_train, 'sec\n')
# Print out the results
prediction = nn.predict_proba(Q_train, train_paramf)
classes = nn.predict(Q_train, train_paramf)
print('\nsyn0\n', train_paramf[0])
print('\nsyn1\n', train_paramf[1])
print('\nClass Probabilities\n', prediction)
print('\nClass\n', classes)
print('\nError\n', cost[-1])

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(range(0, len(cost)), cost)
ax.set_xlabel('Epochs')
ax.set_ylabel('Cost')
plt.show()
plt.clf()

score = MScores(q_trainh, encode_labels(np.array(classes), 10))
print('Precision: ', score.precision())
print('Recall: ', score.recall())

print('****Testing****')
prediction = nn.predict_proba(Q_test, train_paramf)
classes = nn.predict(Q_test, train_paramf)
score_test = MScores(q_testh, encode_labels(np.array(classes), 10))
print('Precision: ', score_test.precision())
print('Recall: ', score_test.recall())
plot_misclass(X_test, y_test, classes)
