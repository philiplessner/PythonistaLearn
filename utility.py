import csv
import numpy as np


class Scaler(object):
            
    def _mr(self, X):
        
        return np.mean(X, axis=0), np.std(X, axis=0)
    
    def fit(self, X):
        '''
        Calculate mean and range for each feature
        Parameter
        X: matrix of j features and i observations for each feature
        >>> X = np.array([1., 2., 3., 4., 5., 6.]).reshape(3, 2)
        >>> scale = Scaler()
        >>> f = scale._mr(X)
        >>> w = (np.array([ 3.,  4.]), np.array([ 1.63299316,  1.63299316]))
        >>> np.allclose(f[0], w[0])
        True
        >>> np.allclose(f[1], w[1])
        True
        '''
        self.stats = self._mr(X)

    def transform(self, X):
        '''
        Normalize each feature u = (xj - mean)/range
        '''
        mdev = np.subtract(X, self.stats[0])
        return np.divide(mdev, self.stats[1])
        
    def denormalize(self, h_theta):
        first = np.array([h_theta[0] - np.dot(h_theta[1:],
                                             np.divide(self.stats[0],
                                                       self.stats[1]))])
        rest = np.array(np.divide(h_theta[1:], self.stats[1]))
        return np.concatenate((first, rest))


def prepend_x0(X):
    '''
    Prepends 1 to each row of a matrix
    >>> Z = np.array([3., 4., 5., 6.]).reshape(2, 2)
    >>> Q = prepend_x0(Z)
    >>> W = np.array([[ 1.,  3.,  4.], [ 1.,  5.,  6.]])
    >>> np.allclose(Q, W)
    True
    '''
    ones = np.array([np.ones(X.shape[0])])
    return np.concatenate((ones.T, X), axis=1)
    

def csv_reader(fpath, Xcols, ycol):
    with open(fpath, 'rU') as f:
        X = []
        y = []   
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[col]) for col in Xcols])
            y.append(float(row[ycol]))
    return X, y

