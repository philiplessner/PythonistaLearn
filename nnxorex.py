from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from toolz import take, iterate
import NNplay as nn


# X is a nexamples x nfeatures matrix
# syn0 is a nfeatures x nhidden matrix of weights
# l1 is a nexamples x nhidden matrix
# syn1 is a nhidden x 1 matrix
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])
nfeatures = X.shape[1]
nexamples = X.shape[0]
nhidden = 16
noutputs = y.shape[1]

syn0, l1, syn1, l2 = nn.initialize(nexamples, nfeatures, nhidden, noutputs)
network0 = (X, syn0, l1, syn1, l2)
alpha = 10.0
        
out = list(take(20000, iterate(partial(nn.fit, y, alpha), network0)))
print(out[-1])
print(nn.error(y, out[-1]))
errors = [nn.error(y, o) for o in out]
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(range(0, len(out)), errors)
plt.show()
plt.clf()

