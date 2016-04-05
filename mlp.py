# /home/jay/anaconda3/bin/python

import numpy as np
from sklearn.metrics import log_loss


sig = lambda x: 1.0/(1.0+np.exp(-x))
sig_d = lambda x: sig(x) * (1 - sig(x))
sig_to_d = lambda x: x * (1 - x)
# log_loss = lambda y,yhat: np.sum(-(y*np.log(yhat) + (1 - y)*np.log(1-yhat)))

def print_shape(f):
    print(list(map(lambda x: x.shape, f)))

def get_shape(f):
    return str(list(map(lambda x: x.shape, f)))

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


        self.biases = [np.random.randn(x) for x in layer_sizes[1:]]

    def __str__(self):

        out = 'weights:' + get_shape(self.weights)
        out += ', biases:' + get_shape(self.biases)
        return out

    def feed_forward(self, X, dropout=False, dropout_percent=0.5):
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
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            if dropout:
                drop_mat = np.random.binomial(
                                [np.ones(z.shape)], 1-dropout_percent)[0] * (
                                1.0/(1-dropout_percent)
                            )
                a = sig(z)
                print(z.shape)
                z *= drop_mat
                print(z.shape)
                a *= drop_mat

            self.z.append(z)
            self.a.append(a)

        return a

    def back_propigation(self, X, Y, dropout=False, dropout_percent=0.5):
        self.feed_forward(X, dropout, dropout_percent)
        ddw = []
        ddb = []
        delta = -(Y - self.a[-1]) * sig_to_d(self.a[-1])

        for i in reversed(range(len(self.weights))):
            ddw.append(np.dot(self.a[i].T, delta))
            ddb.append(delta)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * sig_to_d(self.a[i])

        return list(reversed(ddw)), [b.sum(axis = 0) for b in reversed(ddb)]

    def gradient_descent(self, X, Y, itterations, batch_size, a = 1e-2,
                         dropout=False, dropout_percent=0.5):
        '''
        Parameters
        ----------

        Returns
        -------
        '''
        assert batch_size <= X.shape[0]
        assert X.shape[0] == Y.shape[0], 'X and Y different lengths'

        n_batches = X.shape[0] // batch_size
        batches = np.array_split(range(X.shape[0]), n_batches)

        for i in range(itterations):
            for b in batches:
                ddw, ddb = self.back_propigation(X[b], Y[b],
                                                 dropout, dropout_percent)
                for j in range(len(self.weights)):
                    self.weights[j] -= ddw[j] * a
                    self.biases[j] -= ddb[j] * a

    def predict(self, X):
        '''
            feed_forward without recording activations or using dropout
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
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = sig(z)
        return a

    def gradient_checking(self, X, Y):
        '''
        utility function to check back_propigation
        '''
        bp = self.back_propigation(X, Y)[0]
        yhat = self.feed_forward(X)
        # print_shape(bp)
        weights = self.weights[:]
        epsilon = 1e-4
        # epsilon = 1
        grad_approx = [np.zeros(w.shape) for w in self.weights]

        for i in range(len(self.weights)):

            for j, x in np.ndenumerate(self.weights[i]):
                zeros_mat = np.zeros(self.weights[i].shape)
                # theta_plus
                zeros_mat[j] += epsilon
                self.weights[i] = self.weights[i] + zeros_mat
                p1 = np.array_equal(weights[i], self.weights[i])
                theta_plus = self.feed_forward(X)
                self.weights[i] = weights[i][:]
                # theta_minus
                zeros_mat = np.zeros(self.weights[i].shape)
                zeros_mat[j] = epsilon
                self.weights[i] = self.weights[i] - zeros_mat
                p2 = np.array_equal(weights[i], self.weights[i])
                theta_minus = self.feed_forward(X)
                # reset weights
                self.weights[i] = weights[i][:]
                p3 = np.array_equal(weights[i], self.weights[i])

                if any([p1, p2,  not p3]):
                    print(p1, p2, p3)
                    print(i, j)

                # print((log_loss(Y, theta_plus),
                #       log_loss(Y, theta_minus)), (2 * epsilon))
                grad_approx[i][j] = (log_loss(Y, theta_plus) -
                      log_loss(Y, theta_minus)) / (2 * epsilon)

        # for i,j in zip(bp, grad_approx):
        #     print(i - j)

        return grad_approx



if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import classification_report

    X, Y = load_digits()['data'], load_digits()['target']

    onehot = OneHotEncoder(sparse = False)
    Y = onehot.fit_transform(np.atleast_2d(Y).T)

    train_x, test_x, train_y, test_y = train_test_split( X, Y,
        test_size = 0.3)

    test_y = np.argmax(test_y, 1)
    train_y2 = np.argmax(train_y, 1)

    nn = mlp([X.shape[1], 100, 100, 10])
    yhat = np.argmax(nn.feed_forward(train_x), 1)
    # nn.gradient_descent(train_x, train_y, 10, 100, a = 1e-1)
    # nn.gradient_descent(train_x, train_y, 9000, 100, a = 1e-2)
    nn.gradient_descent(train_x, train_y, 1000, 100, a = 1e-4, dropout = True)
    yhat = np.argmax(nn.feed_forward(train_x), 1)
    print(nn)
    print(classification_report(train_y2, yhat))




    # nn = mlp([2, 2, 1])
    #
    # X = np.array([[10, 10]])
    #             #   [0,1],
    #             #   [1,0],
    #             #   [1,1]])
    #
    # Y = np.array([[0]])
    # 			#   [1],
    # 			#   [1],
    # 			#   [0]])
    # #
    # grad_approx = nn.gradient_checking(X, Y)
    # bp = nn.back_propigation(X,Y)
    # for i,j in zip(grad_approx, bp[0]):
    #     print(i, j, sep =  '\n\n')
    #     print('\n', 'difference:')
    #     print(i - j, '\n\n')
    # # print(grad_approx[-1])
    # # first_dd = nn.simple_bp(X, Y)
    # # print(first_dd)
    # # print(grad_approx[-1], first_dd)
    # # print(grad_approx[-1] - first_dd)
    # # nn.back_propigation(X, Y)
