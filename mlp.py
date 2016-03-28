# /home/jay/anaconda3/bin/python

import numpy as np


sig = lambda x: 1.0/(1.0+np.exp(-x))
sig_d = lambda x: sig(x) * (1 - sig(x))
sig_to_d = lambda x: x * (1 - x)

def print_shape(f):
    print(list(map(lambda x: x.shape, f)))

class mlp:

    def __init__ (self, layer_sizes):
        '''
        Parameters
        ----------
        layer_sizes : list of integers
            sizes of the layres in the network. first number is the input layer
            last number is the output layer
        '''

        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]


    def feed_forward(self, X):
        '''
        Parameters
        ----------
        X : numpy.array shape(records, features)

        '''
        assert X.shape[1] == self.weights[0].shape[0], 'input X is wrong shape'
        a = X
        self.z = []
        self.a = [a]
        for l in self.weights:
            z = np.dot(a, l)
            a = sig(z)
            self.z.append(z)
            self.a.append(a)

        return a

    def back_propigation(self, X, Y):
        yhat = self.feed_forward(X)
        ddw = []
        delta = -((Y - self.a[-1]) * sig_d(self.z[-1]))

        for i in reversed(range(len(self.weights))):
            ddw.append(np.dot(delta, self.a[i + 1].T))
            delta = np.dot(delta, self.weights[i].T) * sig_to_d(self.a[i])

        return ddw

    def gradient_checking(ddw, X):
        return


nn = mlp([3, 6, 2])

X = np.array([[1, 0, 0],
              [2, 1, 1],
              [4, 1, 1],
              [1, 1, 1],
              [0, 0, 0]])

Y = np.atleast_2d(X[:,1:3])

print(nn.back_propigation(X, Y))
