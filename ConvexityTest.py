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
from LayerBuilder import*

print "starting convexity test.."


def CompareLayers(Layer1, Layer2):
    diff = 0
    for ind in Layer1.keys():
        Node1 = Layer1[ind]
        Node2 = Layer2[ind]
        diff += abs(Node1['a'] - Node2['a'])
        diff += abs(Node1['b'] - Node2['b'])
        diff += sum(abs(Node1['w'] - Node2['w']))

    return diff

# import some data to play with

iris = datasets.load_boston()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

print 'fitting scalers...tranforming data...'
X_train, X_train_scaler = Preprocess(X_train)
X_test, X_test_scaler = Preprocess(X_test)
Y_train, Y_train_scaler = Preprocess(Y_train)
Y_test, Y_test_scaler = Preprocess(Y_test)

K = 1
iters = 2000
print 'building layer1...'
Layer1 = BuildLayer(NumNodes=K, X_train=X_train, Y_train=Y_train, n_iter=iters,
                    alpha=0.15, epsilon=0.1, NodeCorrection=False,
                    BoostDecay=True, UltraBoosting=True, threshold=-0.0002)

print 'building layer2...'
Layer2 = BuildLayer(NumNodes=K, X_train=X_train, Y_train=Y_train, n_iter=iters,
                    alpha=0.15, epsilon=0.1, NodeCorrection=False,
                    BoostDecay=True, UltraBoosting=True, threshold=-0.0002)

print 'total difference between layers: ', CompareLayers(Layer1, Layer2)
