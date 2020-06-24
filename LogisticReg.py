import numpy as np
from numpy import array
import csv

X = []
Y = []

with open('data6000.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        X.append([float(x) for x in row[2:]])
        Y.append(row[1])
X = np.array(X)
Y = (array(Y) == 'M').astype('float')
Y = np.expand_dims(Y, -1)#expand the dimention of array -1

def train_test_split(X, Y, split=0.2):
    indices = np.random.permutation(X.shape[0])
    split = int(split * X.shape[0])
    
    train_indices = indices[split:]
    test_indices = indices[:split]

    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = Y[train_indices], Y[test_indices]

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = train_test_split(X, Y)


class LogisticRegression:
    def __init__(self, lr=0.02, n_iter=10000):
        self.lr = lr
        self.n_iter = n_iter

    def predict(self, X):
        X = self.normalize(X)
        linear = self._linear(X)
        preds = self._non_linear(linear)
        return (preds >= 0.5).astype('int')

    def _non_linear(self, X):
        return 1 / (1 + np.exp(-X))

    def _linear(self, X):
        return np.dot(X, self.weights) + self.bias

    def initialize_weights(self, X):
        self.weights = np.random.rand(X.shape[1], 1)
        #add a bias to the terms that
        self.bias = np.zeros((1,))

    def fit(self, X_train, Y_train):
        self.initialize_weights(X_train)
        self.x_mean = X_train.mean(axis=0).T
        self.x_stddev = X_train.std(axis=0).T

        # normalize
        X_train = self.normalize(X_train)

        # Run gradient descent for n iterations
        for i in range(self.n_iter):
            # make normalized predictions
            probs = self._non_linear(self._linear(X_train))
            diff = probs - Y_train

            delta_weights = np.mean(diff * X_train, axis=0, keepdims=True).T
            delta_bias = np.mean(diff)
            # update-weights
            self.weights = self.weights - self.lr * delta_weights
            self.bias = self.bias - self.lr * delta_bias
        return self

    def normalize(self, X):
        X = (X - self.x_mean) / self.x_stddev
        return X
    
    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def loss(self, X, y):
        probs = self._non_linear(self._linear(X))

        # entropy when true class is positive
        pos_log = y * np.log(probs + 1e-15)
        # entropy when true class is negative
        neg_log = (1 - y) * np.log((1 - probs) + 1e-15)

        l = -np.mean(pos_log + neg_log)
        return l

lr = LogisticRegression()
lr.fit(x_train, y_train)
print('Accuracy on test set: {:.2f}%'.format(lr.accuracy(x_test, y_test) * 100))
print('Loss on test set: {:.2f}'.format(lr.loss(x_test, y_test)))
