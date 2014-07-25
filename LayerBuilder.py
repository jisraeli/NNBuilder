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
    BadNode = False
    BadNodeInfo = [BadNode, 0, 0]
    g_total = 0.0
    for ind in Layer.keys():
        Node = Layer[ind]
        predict = Node['predict']
        a = Node['a']
        S = predict(X_train) / a
        g = numpy.dot(Y_pseudo, S) / len(Y_pseudo)
        g_total += g
        print "g: ", g
        p = 1.0
        lam = g / p
        if lam * a < threshold:
            if abs(lam) > abs(BadNodeInfo[1]):
                BadNode = True
                BadNodeInfo = [BadNode, lam, ind]
        elif not BadNode:
            if abs(lam) > abs(BadNodeInfo[1]):
                BadNode = False  # redundant?
                BadNodeInfo = [BadNode, lam, ind]

    return [BadNodeInfo, Y_pseudo, g_total]


def BoostNodes(Layer, X_train, Y_train, epsilon=0.01, g_tol=0.01,
               threshold=-0.01):
    '''
    boosts/correct node until therhold or until a node is trapped
    '''
    sign = lambda x: math.copysign(1, x)
    [BadNode, lam, ind], _, _ = CheckLayer(Layer, X_train, Y_train,
                                           threshold=threshold)
    lam_prev, ind_prev = [0, 0]
    N = 1.0*len(Layer.keys())
    while lam > (g_tol / N) or BadNode:
        lam_prev, ind_prev = [lam, ind]
        [BadNode, lam, ind], _, _ = CheckLayer(Layer, X_train, Y_train,
                                               threshold=threshold)
        if ind==ind_prev and sign(lam)!=sign(lam_prev):
            print "Node is Trapped! Stopping Current Boosting!"
            break
        elif BadNode:  # check if theres g<0, then correct
            print "correcting node: ", ind
            Node = Layer[ind]
            Node['lr'] += epsilon * sign(lam) / Node['a']
            print "Layer boost weights :", PrintRates(Layer)
        elif not BadNode:
            print "boosting node: ", ind
            Node = Layer[ind]
            Node['lr'] += epsilon * sign(lam) / Node['a']
            print "Layer boost weights :", PrintRates(Layer)


def EvalNode(Node, X_train, Y_pseudo):
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
               NodeCorrection=True, BoostDecay=False, UltraBoosting=False,
               g_tol=0.01, threshold=-0.01):
    Layer = InitLayer(X_train, Y_train, n_iter, alpha, epsilon=epsilon)
    i = 0
    while i < NumNodes:
        if NodeCorrection:  # check if nodes should be corrected/boosted
            BoostNodes(Layer=Layer, X_train=X_train, Y_train=Y_train,
                       epsilon=epsilon, g_tol=g_tol, threshold=threshold)
            [_, lam, ind], Y_pseudo, _ = CheckLayer(Layer, X_train, Y_train,
                                                    threshold=threshold)
            Node = NewNode(Layer, X_train, Y_train, n_iter=n_iter, alpha=alpha,
                           epsilon=epsilon)
            if EvalNode(Node, X_train, Y_pseudo) > lam:  # > best node?
                print "adding node number ", i+2
                AddNode(Layer, Node)
                i += 1
                if BoostDecay:
                    epsilon = epsilon * i / (i+1)
            else:
                print "New node after boosting is not good enough!"
                g_tol = g_tol / 2.0
                print "reducing g_tol to: ", g_tol
        else:
            print "adding node number ", i+2
            Node = NewNode(Layer, X_train, Y_train, n_iter=n_iter, alpha=alpha,
                           epsilon=epsilon)
            AddNode(Layer, Node)
            print "Layer boost weights :", PrintRates(Layer)
            i += 1
    if NodeCorrection and UltraBoosting:
        print "starting UltraBoosting..."
        for t in range(2):
            BoostNodes(Layer=Layer, X_train=X_train, Y_train=Y_train,
                       epsilon=epsilon, g_tol=g_tol, threshold=threshold)
            print "------------Finished Ultra Boost with g_tol="+str(g_tol),
            print "-------------"
            g_tol = g_tol / 2.0

    return Layer


def Preprocess(X):
    '''
    PreProcesses data arrays
    returns
        transformed array X
        fitted scaler
    '''
    if len(numpy.shape(X)) == 2:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    elif len(numpy.shape(X)) == 1:
        X = numpy.reshape(X, (len(X), 1))
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        X = numpy.reshape(X, (len(X)))

    return [X, scaler]


def Postprocess(X, scaler):
    '''
    Inverse transforms X using scaler
    '''
    if len(numpy.shape(X)) == 2:
        X = scaler.inverse_transform(X)
    elif len(numpy.shape(X)) == 1:
        X = numpy.reshape(X, (len(X), 1))
        X = scaler.inverse_transform(X)
        X = numpy.reshape(X, (len(X)))

    return X
