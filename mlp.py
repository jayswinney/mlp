# /home/jay/anaconda3/bin/python

import numpy as np


sig = lambda x: 1.0/(1.0+np.exp(-x))
sig_d = lambda x: sig(x) * (1 - sig(x))
sig_to_d = lambda x: x * (1 - x)
log_loss = lambda y, yhat: np.sum(-(y*np.log(yhat) + (1 - y)*np.log(1 - yhat)))

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
        np.random.seed(1)
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]


    def feed_forward(self, X):
        '''
        Parameters
        ----------
        X : numpy.array shape(records, features)

        Returns
        -------
        a : numpy.array
            output of neural network

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
        self.feed_forward(X)
        ddw = []
        delta = -((Y - self.a[-1]) * sig_d(self.z[-1]))
        print ('delta.shape:', delta.shape)

        for i in reversed(range(len(self.weights))):
            ddw.append(np.dot(self.a[i].T, delta))
            delta = np.dot(delta, self.weights[i].T) * sig_to_d(self.a[i])
            print ('delta %s shape:' % str(i), delta.shape)

        print_shape(list(reversed(ddw)))
        print_shape(self.weights)
        return list(reversed(ddw))

    def gradient_checking(self, X, Y):
        '''
        utility function to check back_propigation
        '''
        bp = self.back_propigation(X, Y)
        print_shape(bp)
        weights = self.weights[:]
        epsilon = 1e-4

        grad_approx = [np.zeros(w.shape) for w in self.weights]

        for i in range(len(self.weights)):

            for j,h in zip(np.nditer(grad_approx[i], op_flags=['readwrite']),
                           np.nditer(self.weights[i], op_flags=['readwrite'])):
                h += epsilon
                theta_plus = self.feed_forward(X)
                h -= 2*epsilon
                theta_minus = self.feed_forward(X)
                h += epsilon
                j = (log_loss(Y, theta_plus) - log_loss(Y, theta_minus)) / \
                        (2 * epsilon)

        for i,j in zip(bp, grad_approx):
            print(i - j)


nn = mlp([3, 4, 1])

X = np.array([[0,0,1]])
            #   [0,1,1],
            #   [1,0,1],
            #   [1,1,1]])

Y = np.array([[0]])
			#   [1],
			#   [1],
			#   [0]])
#
print(nn.gradient_checking(X, Y))
# nn.back_propigation(X, Y)
