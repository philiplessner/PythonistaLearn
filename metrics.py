import numpy as np
from toolz import pluck
from linreg import errors
    

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum((y - np.mean(y))**2)
    

def r2(X, y, h_theta):
    sum_of_squared_errors = sum(errors(X, y, h_theta)**2)
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(np.array(y))


class Scores(object):
    def __init__(self, y, yp):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for yi, ypi in zip(y, yp):
            if yi == 1 and ypi == 1:
                true_positives += 1
            elif yi == 1 and ypi == 0:
                false_negatives += 1
            elif yi == 0 and ypi == 1:
                false_positives += 1
            else:
                true_negatives += 1
        self.tp = true_positives
        self.fn = false_negatives
        self.fp = false_positives
        self.tn = true_negatives
    
    def accuracy(self):
        correct = self.tp + self.tn
        total = self.tp + self.fp + self.fn + self.tn
        return correct / total
    
    def precision(self):
        return self.tp / (self.tp + self.fp)
        
    def recall(self):
        return self.tp / (self.tp + self.fn)
        
    def f1_score(self):
       return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
       
       
class MScores (object):
    def __init__(self, y, yp):
        self.tp = []
        self.fn = []
        self.fp = []
        self.tn = []
        self.nclasses = len(yp[0])
        for nclass in range(0, len(yp[0])):
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            for yi, ypi in zip(pluck(nclass, y), pluck(nclass, yp)):
                if yi == 1 and ypi == 1:
                    true_positives += 1
                elif yi == 1 and ypi == 0:
                    false_negatives += 1
                elif yi == 0 and ypi == 1:
                    false_positives += 1
                else:
                    true_negatives += 1
            self.tp.append(true_positives)
            self.fn.append(false_negatives)
            self.fp.append(false_positives)
            self.tn.append(true_negatives)
            
    def precision(self):      
        return sum(self.tp[nclass] / (self.tp[nclass] + self.fp[nclass]) for nclass in range(self.nclasses)) / self.nclasses
        
    def recall(self):
        return sum(self.tp[nclass] / (self.tp[nclass] + self.fn[nclass]) for nclass in range(self.nclasses)) / self.nclasses
        
