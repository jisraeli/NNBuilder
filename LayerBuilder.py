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
    Node['lr'] = epsilon
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
        pred += predict(X_train) * node['lr']
    Y_pseudo = Y_train - pred
    Node = OptimalNode(X_train, Y_pseudo, bias=True, n_iter=n_iter,
                       alpha=alpha)
    Node['lr'] = epsilon

    return Node


def AddNode(Layer, Node):
    NodeNumber = len(Layer.keys()) + 1
    Layer[str(NodeNumber)] = Node


def CheckLayer(Layer, X_train, Y_train, threshold=-0.01):
    '''
    IMPORTANT!
    inputs
        existing layer
    outputs
        nodes that need to be corrected
    '''
    pred = 0
    for ind in Layer.keys():
            node = Layer[ind]
            predict = node['predict']
            pred += predict(X_train) * node['lr']
    Y_pseudo = Y_train - pred
    BadNodeInfo = [False, 0, 0]
    g_total = 0.0
    for ind in Layer.keys():
        Node = Layer[ind]
        predict = Node['predict']
        a = Node['a']
        S = predict(X_train) / a
        g = numpy.dot(Y_pseudo, S) / len(Y_pseudo)
        g_total += g
        #print "g: ", g
        p = 1.0
        lam = g / p
        if lam * a < threshold:
            if abs(lam) > abs(BadNodeInfo[1]):
                BadNodeInfo = [True, lam, ind]
        else:
            if abs(lam) > abs(BadNodeInfo[1]):
                BadNodeInfo = [False, lam, ind]

    return [BadNodeInfo, Y_pseudo, g_total]


def EvalNode(Node, Y_pseudo):
    predict = Node['predict']
    a = Node['a']
    S = predict(X_train) / a
    g = numpy.dot(Y_pseudo, S) / len(Y_pseudo)
    p = 1.0
    lam = g / p

    return lam


def PrintRates(Layer):
    '''
    prints the boosting weights of each node
    '''
    lrList = []
    for ind in range(len(Layer.keys())):
        node = Layer[str(ind+1)]
        lrList.append(node['lr'] * node['a'])
    print lrList


def BuildLayer(NumNodes, X_train, Y_train, n_iter, alpha, epsilon=0.01,
               NodeCorrection=True, BoostDecay=False, UltraBoosting=False):
    Layer = InitLayer(X_train, Y_train, n_iter, alpha, epsilon=epsilon)
    sign = lambda x: math.copysign(1, x)
    i = 0
    if UltraBoosting:
        threshold = -0.1
    while i < NumNodes + 1:
        if NodeCorrection:
            [BadNode, lam, ind], Y_pseudo, g_total = CheckLayer(Layer, X_train, Y_train, threshold=threshold)
            print "Bad node: ", BadNode
            if BadNode:
                print "correcting node: ", ind
                Node = Layer[ind]
                Node['lr'] += epsilon * sign(lam) / Node['a']
                print "Layer boost weights :", PrintRates(Layer)
            else:
                Node = NewNode(Layer, X_train, Y_train, n_iter=n_iter,
                               alpha=alpha, epsilon=epsilon)
                if EvalNode(Node, Y_pseudo) > lam:
                    if i < NumNodes:
                        print "adding node number ", i+2
                        AddNode(Layer, Node)
                    i += 1
                    if BoostDecay:
                        epsilon = epsilon * i / (i+1)
                else:
                    print "boosting node: ", ind
                    Node = Layer[ind]
                    Node['lr'] += epsilon * sign(lam) / Node['a']
                    print "Layer boost weights :", PrintRates(Layer)
        else:
            print "adding node number ", i+2
            Node = NewNode(Layer, X_train, Y_train, n_iter=n_iter, alpha=alpha,
                           epsilon=epsilon)
            AddNode(Layer, Node)
            print "Layer boost weights :", PrintRates(Layer)
            i += 1
    if NodeCorrection and UltraBoosting:
        print "starting UltraBoosting..."
        epsilon *= i + 1
        for t in range(1, 4):
            while g_total > 0.01:
                [BadNode, lam, ind], Y_pseudo, g_total = CheckLayer(Layer, X_train, Y_train)
                #print "Bad node: ", BadNode
                if BadNode:
                    #print "correcting node: ", ind
                    Node = Layer[ind]
                    Node['lr'] += epsilon * sign(lam) / Node['a']
                    #print "Layer boost weights :", PrintRates(Layer)
                else:
                    #print "boosting node: ", ind
                    Node = Layer[ind]
                    Node['lr'] += epsilon * sign(lam) / Node['a']
                    #print "Layer boost weights :", PrintRates(Layer)
            epsilon *= t / (t + 1)
            print "-------------Finished Ultra Boost"+str(t)+"------------------"


    return Layer


# import some data to play with

iris = datasets.load_boston()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

print 'fitting scalers...'
X_train_scaler = preprocessing.StandardScaler().fit(X_train)
X_test_scaler = preprocessing.StandardScaler().fit(X_test)
Y_train = numpy.reshape(Y_train, (len(Y_train), 1))
Y_train_scaler = preprocessing.StandardScaler().fit(Y_train)
Y_test = numpy.reshape(Y_test, (len(Y_test), 1))
Y_test_scaler = preprocessing.StandardScaler().fit(Y_test)

print 'tranforming data...'
X_train = X_train_scaler.transform(X_train)
X_test = X_test_scaler.transform(X_test)


Y_train = Y_train_scaler.transform(Y_train)
Y_train = numpy.reshape(Y_train, (len(Y_train)))

Y_test = Y_test_scaler.transform(Y_test)
Y_test = numpy.reshape(Y_test, (len(Y_test)))
print "initializing layer.."
Layer = InitLayer(X_train, Y_train, n_iter=500, alpha=0.7)
K = 5
print "building layer..."
Layer = BuildLayer(NumNodes=K, X_train=X_train, Y_train=Y_train, n_iter=500,
                   alpha=0.35, epsilon=0.8, NodeCorrection=True, BoostDecay=True, UltraBoosting=True)
print "number of nodes in layer: ", len(Layer.keys())

pred_train = 0
pred_test = 0
for ind in Layer.keys():
    node = Layer[ind]
    predict = node['predict']
    pred_train += predict(X_train) * node['lr']
    pred_test += predict(X_test) * node['lr']

print "Running Adabost with LR for comparison..."
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
clf = AdaBoostRegressor(base_estimator=LogisticRegression(), n_estimators=K+1,
                        loss='square')
clf.fit(X_train, Y_train)
pred_clf = clf.predict(X_test)

print "Final layer results:"


Y_train = numpy.reshape(Y_train, (len(Y_train), 1))
Y_train = Y_train_scaler.inverse_transform(Y_train)
Y_train = numpy.reshape(Y_train, (len(Y_train)))

pred_train = numpy.reshape(pred_train, (len(pred_train), 1))
pred_train = Y_train_scaler.inverse_transform(pred_train)
pred_train = numpy.reshape(pred_train, (len(pred_train)))

print "Prediction on train data: ", pred_train
print "actual train data: ", Y_train
print "train error: ", 1.0 * sum(abs(Y_train-pred_train)**2) / len(Y_train)

Y_test = numpy.reshape(Y_test, (len(Y_test), 1))
Y_test = Y_test_scaler.inverse_transform(Y_test)
Y_test = numpy.reshape(Y_test, (len(Y_test)))

pred_test = numpy.reshape(pred_test, (len(pred_test), 1))
pred_test = Y_test_scaler.inverse_transform(pred_test)
pred_test = numpy.reshape(pred_test, (len(pred_test)))

print "Prediction on test data: ", pred_test
print "actual test data: ", Y_test
print "test error: ", 1.0 * sum(abs(Y_test-pred_test)**2) / len(Y_test)

pred_clf = numpy.reshape(pred_clf, (len(pred_clf), 1))
pred_clf = Y_test_scaler.inverse_transform(pred_clf)
pred_clf = numpy.reshape(pred_clf, (len(pred_clf)))

err = numpy.mean((pred_clf - Y_test)**2)
print "Scikit's Adaboost with LR on transformed data, test error: ", err

X_train = X_train_scaler.inverse_transform(X_train)
X_test = X_test_scaler.inverse_transform(X_test)

clf = AdaBoostRegressor(base_estimator=LogisticRegression(), n_estimators=K+1,
                        loss='square')
clf.fit(X_train, Y_train)
pred_clf = clf.predict(X_test)
err = numpy.mean((pred_clf - Y_test)**2)
print "Scikit's Adaboost with LR on original data, test error: ", err