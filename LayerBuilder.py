import numpy
import theano
import sys
import theano.tensor as T
from theano import pp
from sklearn import datasets
from NodeOptimize import OptimalNode
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import math
import IPython


def InitLayer(X_train, Y_train, n_iter, alpha, epsilon=1.0):
    '''
    inputs
        x_train: training features
        y_train: response variable
        n_iter: # of iterations for SGD
        alpha: strength of L2 penalty (default penalty for now)
        epsilon: learning rate, scales coefficient of node
    outputs
        Layer: node dictionary containing initial node
    '''
    Layer = {}
    Node = OptimalNode(X_train, Y_train, bias=True, n_iter=n_iter, alpha=alpha)
    Node['a'] *= epsilon
    Layer['1'] = Node

    return Layer


def NewNode(Layer, X_train, Y_train, n_iter=5, alpha=0.01, epsilon=1.0):
    '''
    inputs
        x_train: training features
        y_train: response variable
        n_iter: # of iterations for SGD
        alpha: strength of L2 penalty (default penalty for now)
        epsilon: learning rate, scales coefficient of node
    outputs
        no output, mutates the Layer by adding a new node
    '''
    pred = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred += predict(X_train)
    Y_pseudo = Y_train - pred
    Node = OptimalNode(X_train, Y_pseudo, bias=True, n_iter=n_iter, alpha=alpha)
    Node['a'] *= epsilon
    NodeNumber = len(Layer.keys()) + 1
    Layer[str(NodeNumber)] = Node


def CheckLayer(Layer, X_train, Y_pseudo):
    '''
    IMPORTANT!
    inputs
        existing layer
    outputs
        nodes that need to be corrected
    '''
    BadNodeInfo = [False, 0, 0]
    for ind in Layer.keys():
        Node = Layer[ind]
        predict = Node['predict']
        a = Node['a']
        S = predict(X_train) / a
        g = numpy.dot(Y_pseudo, S) / len(Y_pseudo)
        p = 1.0
        lam = g / p
        if lam*a < 0:
            if abs(lam) > abs(BadNodeInfo[1]):
                BadNodeInfo = [True, lam, ind]

    return BadNodeInfo


def BuildLayer(NumNodes, X_train, Y_train, n_iter, alpha, epsilon=0.01):
    Layer = InitLayer(X_train, Y_train, n_iter, alpha, epsilon=0.01)
    sign = lambda x: math.copysign(1, x)
    pred = 0
    i = 0
    while i < NumNodes:
        for ind in Layer.keys():
            node = Layer[ind]
            predict = node['predict']
            pred += predict(X_train)
        Y_pseudo = Y_train - pred
        BadNode, lam, ind = CheckLayer(Layer, X_train, Y_pseudo)
        if BadNode:
            print "correcting node: ", ind
            Node = Layer[ind]
            Node['a'] += epsilon * sign(lam)
        else:
            print "adding node number ", i+2
            NewNode(Layer, X_train, Y_train, n_iter=n_iter, alpha=alpha)
            i += 1

    return Layer


# import some data to play with
#iris = datasets.load_iris()
iris = datasets.load_boston()
#X = iris.data[:, :4]  # we only take the first two features.
X = iris.data
Y = iris.target
min_max_scaler = preprocessing.MinMaxScaler()
print "shaoe if X: ", numpy.shape(X)
print "shape of Y: ", numpy.shape(Y)
X = min_max_scaler.fit_transform(X)
N = len(Y)
Y = numpy.reshape(Y, (N, 1))
Y = min_max_scaler.fit_transform(Y)
Y = numpy.reshape(Y, (N))
#inds = Y<2
#Y = Y[inds]
#X = X[inds, :]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print "initializing layer.."
Layer = InitLayer(X_train, Y_train, n_iter=500, alpha=0.7)
N = 8
print "building layer..."
Layer = BuildLayer(NumNodes=N, X_train=X_train, Y_train=Y_train, n_iter=500,
                   alpha=0.7, epsilon=0.2)
'''
for i in range(N):
    NewNode(Layer, X_train, Y_train, n_iter=500, alpha=0.7)
'''
print "number of nodes in layer: ", len(Layer.keys())

pred_train = 0
pred_test = 0
"printing predictions on first 5 trainig samples:"
for ind in Layer.keys():
    node = Layer[ind]
    predict = node['predict']
    #print "node ", ind, predict(X_train[:5, :])
    pred_train += predict(X_train)
    pred_test += predict(X_test)
#sys.exit()
print "Final layer results:"


print "Prediction on train data: ", pred_train
print "actual train data: ", Y_train
print "train error: ", 1.0 * sum(abs(Y_train-pred_train)**2) / len(Y_train)


print "Prediction on test data: ", pred_test
print "actual test data: ", Y_test
print "test error: ", 1.0 * sum(abs(Y_test-pred_test)**2) / len(Y_test)