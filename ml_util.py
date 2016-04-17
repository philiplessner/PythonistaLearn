import random
import numpy as np
from toolz import compose
from utility import Scaler


def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    random.seed(1)
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


def train_test_split(data, test_pct):
    # split the dataset of pairs 
    train_data, test_data = split_data(data, 1 - test_pct)   
    return train_data, test_data


def scale_data(train_data, test_data):
    Z_train, y_train = zip(*train_data)       
    scale = Scaler()
    scale.fit(Z_train)
    transform = compose(prepend_x0, scale.transform)
    scaledX_train = transform(Z_train)
    scaled_train = list(zip(scaledX_train, y_train))
    Z_test, y_test = zip(*test_data)
    scaledX_test = transform(Z_test)
    scaled_test = list(zip(scaledX_test, y_test))
    return scaled_train, scaled_test
    

def encode_labels(y, noutputs):
    onehot = np.zeros((y.shape[0], noutputs)) 
    for irow, val in enumerate(y):
         onehot[irow, val] = 1.0
    return onehot

