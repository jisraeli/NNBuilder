import numpy
import theano
import sys
import theano.tensor as T
from theano import pp
from sklearn import datasets
from NodeOptimize import OptimalNode, EarlyStopNode
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import math
import IPython
import timeit
import matplotlib.pyplot as plt
from copy import deepcopy


def InitLayer(X_train_node, Y_train_node, X_validate_node, Y_validate_node,
              n_iter, alpha, epsilon=1.0, minibatch=False, nodeCV_size=0.1):
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
    
    Node = OptimalNode(X_train_node, Y_train_node, bias=True, n_iter=n_iter,
                       alpha=alpha, minibatch=minibatch)
    Node = EarlyStopNode(Node, X_validate_node, Y_validate_node)

    Node['lr'] = epsilon
    Layer['1'] = Node

    return Layer


def NewNode(Layer, X_train_node, Y_train_node, X_validate_node,
            Y_validate_node, n_iter=5, alpha=0.01, epsilon=1.0,
            minibatch=False):
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
    pred_train = 0
    pred_validate = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred_train += predict(X_train_node) * node['lr']
        pred_validate += predict(X_validate_node) * node['lr']
    Y_pseudo = Y_train_node - pred_train
    y_pseudo_validate = Y_validate_node - pred_validate

    Node = OptimalNode(X_train_node, Y_pseudo, bias=True, n_iter=n_iter,
                       alpha=alpha, minibatch=minibatch)
    Node = EarlyStopNode(Node, X_validate_node, y_pseudo_validate)
    Node['lr'] = epsilon

    return Node


def AddNode(Layer, Node, X_validate, Y_validate):
    pred_validate = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred_validate += predict(X_validate) * node['lr']
    err_before_node = numpy.mean(abs(Y_validate - pred_validate)**1)
    predict = Node['predict']
    pred_validate += predict(X_validate) * node['lr']
    err_after_node = numpy.mean(abs(Y_validate - pred_validate)**1)
    UsefulNode = False
    print "err before node: ", err_before_node
    print "err with node: ", err_after_node
    if err_after_node < err_before_node:
        UsefulNode = True
        NodeNumber = len(Layer.keys()) + 1
        Layer[str(NodeNumber)] = Node

    return UsefulNode


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
        #print "g: ", g
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


def UsefulNode(Layer, NewNode, X_validate_layer, Y_validate_layer):
    pred_validate = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred_validate += predict(X_validate_layer) * node['lr']
    err_Layer = numpy.mean(abs(Y_validate_layer - pred_validate)**2)
    pred_new = NewNode['predict']
    pred_withNode = pred_validate + pred_new(X_validate_layer) * NewNode['lr']
    err_withNode = numpy.mean(abs(Y_validate_layer - pred_withNode)**2)
    AddNode = False
    if err_withNode < err_Layer:
        AddNode = True

    return AddNode


def ExtendLayer(Layer, NewNode):
    N = len(Layer.keys())
    ind = N + 1
    Layer[str(ind)] = NewNode


def PrintRates(Layer):
    '''
    prints the boosting weights of each node
    '''
    lrList = []
    for ind in range(len(Layer.keys())):
        node = Layer[str(ind+1)]
        lrList.append(node['lr'] * node['a'])
    print lrList


def BuildLayer(NumNodes, X_train, Y_train, X_validate_layer, Y_validate_layer,
               n_iter, alpha, epsilon=1.0, BoostDecay=False,
               UltraBoosting=False, g_tol=0.01, g_final=0.0000001,
               threshold=-0.01, minibatch=False, nodeCV_size=0.1):
    '''
    Builds a Layer by optimizing new nodes and adding them if they are useful.
    Here's how it works:
        Calls InitLayer and New Node
            These randomly split training set into node_training and
            node_validation sets
            Optimize new node w.r.t. residuals on node_training
            Choose an EarlyStop using node_validation
            Set 'lr' to 1.0
        Then Calls EvalNode
            ExtendLayer checks Layer's errors on layer_validation set with and
            without new node.
            If node reduces erro:
                Call AddNode and add the new node
            else:
                stop building the layer

    '''
    train_validate = train_test_split(X_train, Y_train, test_size=nodeCV_size)
    [X_train_node, X_validate_node,
        Y_train_node, Y_validate_node] = train_validate
    print 'Initializing Layer..'
    Layer = InitLayer(X_train_node=X_train_node, Y_train_node=Y_train_node,
                      X_validate_node=X_validate_node,
                      Y_validate_node=Y_validate_node, n_iter=n_iter,
                      alpha=alpha, epsilon=epsilon, minibatch=minibatch)
    i = 0
    while i < NumNodes:
        train_validate = train_test_split(X_train, Y_train,
                                          test_size=nodeCV_size)
        [X_train_node, X_validate_node,
            Y_train_node, Y_validate_node] = train_validate
        print 'Optimizing New Node...'
        Node = NewNode(Layer=Layer, X_train_node=X_train_node,
                       Y_train_node=Y_train_node,
                       X_validate_node=X_validate_node,
                       Y_validate_node=Y_validate_node, n_iter=n_iter,
                       alpha=alpha, epsilon=epsilon, minibatch=minibatch)
        AddNode = UsefulNode(Layer=Layer, NewNode=Node,
                             X_validate_layer=X_validate_layer,
                             Y_validate_layer=Y_validate_layer)
        if AddNode:
            print 'Adding Node: ', i+2
            ExtendLayer(Layer=Layer, NewNode=Node)
            i += 1
        else:
            print 'New Node increases validation error - Terminating Layer!'
            break

    return Layer


def Preprocess(X):
    '''
    PreProcesses data arrays
    returns
        transformed array X
        fitted scaler
    '''
    if len(numpy.shape(X)) == 2:
        #scaler = preprocessing.MinMaxScaler().fit(X)
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    elif len(numpy.shape(X)) == 1:
        X = numpy.reshape(X, (len(X), 1))
        #scaler = preprocessing.MinMaxScaler().fit(X)
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


def FoldLabels(Y):

    inds = Y<0
    Y[inds] = -Y[inds]

    return [Y, inds]


def unFoldLabels(Y, inds):
    Y[inds] = -Y[inds]

    return Y


def RunLayerBuilder(NumNodes, X, Y, n_iter, alpha, epsilon=0.01, test_size=0.3,
                    boostCV_size=0.2, nodeCV_size=0.1,
                    BoostDecay=False, UltraBoosting=False, g_final=0.0000001,
                    g_tol=0.01, threshold=-0.01, minibatch=False,
                    SymmetricLabels=False):

    print "creating training, validation, and testing sets..."
    train_test = train_test_split(X, Y, test_size=test_size)
    x_train, X_test, y_train, Y_test = train_test

    print 'fitting scalers...tranforming data...'
    if SymmetricLabels:
        x_train, x_train_inds = FoldLabels(x_train)
        X_test, X_test_inds = FoldLabels(X_test)
    x_train, x_train_scaler = Preprocess(x_train)
    X_test, X_test_scaler = Preprocess(X_test)
    y_train, y_train_scaler = Preprocess(y_train)

    train_validate = train_test_split(x_train, y_train, test_size=boostCV_size)
    X_train, X_validate_layer, Y_train, Y_validate_layer = train_validate

    print 'Running Basic Layer Builder...'
    start = timeit.default_timer()
    Layer = BuildLayer(NumNodes=NumNodes-1, X_train=X_train, Y_train=Y_train,
                       X_validate_layer=X_validate_layer,
                       Y_validate_layer=Y_validate_layer,
                       n_iter=n_iter, alpha=alpha, epsilon=1.0,
                       BoostDecay=False, UltraBoosting=False, g_tol=0.01,
                       g_final=0.0000001, threshold=-0.01, minibatch=False,
                       nodeCV_size=0.1)
    stop = timeit.default_timer()

    print "Layer Building RunTime: ", stop - start
    N = len(Layer.keys())
    print "number of nodes in layer: ", N

    pred_train = 0
    pred_validate = 0
    pred_test = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred_train += predict(X_train) * node['lr']
        pred_validate += predict(X_validate_layer) * node['lr']
        pred_test += predict(X_test) * node['lr']

    # stack training+validation sets, inverse transform, separate again
    K = len(Y_train)
    x_train = numpy.vstack((X_train, X_validate_layer))
    y_train =numpy.hstack((Y_train, Y_validate_layer))

    x_train = Postprocess(x_train, x_train_scaler)
    y_train = Postprocess(y_train, y_train_scaler)
    pred_train = Postprocess(pred_train, y_train_scaler)
    pred_validate = Postprocess(pred_validate, y_train_scaler)
    X_train, X_validate_layer = [x_train[:K, :], x_train[K:, :]]
    Y_train, Y_validate_layer = [y_train[:K], y_train[K:]]

    print "Final layer results:"

    err_train = numpy.mean(abs(Y_train - pred_train)**2)
    print "train error: ", err_train

    err_validate = numpy.mean(abs(Y_validate_layer - pred_validate)**2)
    print "validation error: ", err_validate

    #Y_test = Postprocess(Y_test, y_train_scaler)
    pred_test = Postprocess(pred_test, y_train_scaler)
    err_test = numpy.mean(abs(Y_test - pred_test)**2)
    print "test error: ", err_test

    X_test = Postprocess(X_test, X_test_scaler)
    if SymmetricLabels:
        x_train = unFoldLabels(x_train, x_train_inds)
        X_test = unFoldLabels(X_test, X_test_inds)

    print "Running Adabost, SVM, and LogisticRegression for comparison..."
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    AB = AdaBoostRegressor(loss='square', n_estimators=NumNodes)
    LB = AdaBoostRegressor(base_estimator=LogisticRegression(), loss='square',
                           n_estimators=NumNodes)
    SVM_lin = SVR(kernel='linear')
    SVM_rbf = SVR(kernel='rbf')

    AB.fit(X_train, Y_train)
    LB.fit(X_train, Y_train)
    SVM_lin.fit(X_train, Y_train)
    SVM_rbf.fit(X_train, Y_train)

    err_AB = numpy.mean(abs(AB.predict(X_test) - Y_test)**2)
    err_LB = numpy.mean(abs(LB.predict(X_test) - Y_test)**2)
    err_SVM_lin = numpy.mean(abs(SVM_lin.predict(X_test) - Y_test)**2)
    err_SVM_rbf = numpy.mean(abs(SVM_rbf.predict(X_test) - Y_test)**2)

    print "Scikit's Adaboost on original data, test error: ", err_AB
    print "Scikit's LB on original data, test error: ", err_LB
    print "Scikit's linear SVR on original data, test error: ", err_SVM_lin
    print "Scikit's gaussian SVR on original data, test error: ", err_SVM_rbf

    errs = [err_train, err_validate, err_test,
            err_AB, err_LB, err_SVM_lin, err_SVM_rbf]
    #results = [X_test, Y_test, pred_test]

    return [errs, N]
