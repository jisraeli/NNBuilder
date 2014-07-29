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


def InitLayer(X_train, Y_train, X_validate, Y_validate, n_iter, alpha,
              epsilon=1.0, minibatch=False, nodeCV_size=0.1):
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

    #train_validate = train_test_split(X_train, Y_train, test_size=nodeCV_size)
    #x_train, x_validate, y_train, y_validate = train_validate

    Node = OptimalNode(X_train, Y_train, bias=True, n_iter=n_iter, alpha=alpha,
                       minibatch=minibatch)

    #print "----training losses----"
    #Node = EarlyStopNode(Node, x_train, y_train)
    #print "----validation losses----"
    Node = EarlyStopNode(Node, X_validate, Y_validate)
    #sys.exit()

    Node['lr'] = epsilon
    Layer['1'] = Node

    return Layer


def NewNode(Layer, X_train, Y_train, X_validate, Y_validate, n_iter=5,
            alpha=0.01, epsilon=1.0, minibatch=False, nodeCV_size=0.1):
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
        pred_train += predict(X_train) * node['lr']
        pred_validate += predict(X_validate) * node['lr']
    Y_pseudo = Y_train - pred_train
    Y_pseudo_validate = Y_validate - pred_validate

    #train_validate = train_test_split(X_train, Y_pseudo, test_size=nodeCV_size)
    #x_train, x_validate, y_pseudo, y_pseudo_validate = train_validate

    Node = OptimalNode(X_train, Y_pseudo, bias=True, n_iter=n_iter,
                       alpha=alpha, minibatch=minibatch)

    Node = EarlyStopNode(Node, X_validate, Y_pseudo_validate)

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


def BoostedNodes(Layer, X_train, Y_train, epsilon=0.01, g_tol=0.01,
                 threshold=-0.01):
    '''
    boosts/correct node until therhold or until a node is trapped
    '''
    # store lr values before boosting path
    lr_List = []
    for ind in Layer.keys():
        Node = Layer[ind]
        lr_List.append(Node['lr'])

    # run boosting path on training set
    Updates = []
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
            Node = Layer[ind]
            Updates.append([ind, epsilon * sign(lam) / Node['a']])
            Node['lr'] += epsilon * sign(lam) / Node['a']
        elif not BadNode:
            Node = Layer[ind]
            Updates.append([ind, epsilon * sign(lam) / Node['a']])
            Node['lr'] += epsilon * sign(lam) / Node['a']

    # reset Layer lr values to lr_list
    for ind in range(len(lr_List)):
        Node = Layer[str(ind+1)]
        Node['lr'] = lr_List[ind]

    return Updates


def ValidatedNodes(Layer, X_train, Y_train, X_validate, Y_validate,
                   epsilon=0.01, g_tol=0.01, threshold=-0.01):

    Updates = BoostedNodes(Layer=Layer, X_train=X_train, Y_train=Y_train,
                           epsilon=epsilon, g_tol=g_tol, threshold=threshold)
    print "validating ", len(Updates), "boost/correct updates..."
    pred_validate = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred_validate += predict(X_validate) * node['lr']
    err_validate_best = numpy.mean(abs(Y_validate - pred_validate)**2)
    update_ind = 0
    update_best = 0
    for update in Updates:
        ind, lr_update = update
        node = Layer[ind]
        predict = node['predict']
        pred_validate += predict(X_validate) * lr_update
        err_validate = numpy.mean(abs(Y_validate - pred_validate)**2)
        if err_validate < err_validate_best:
            err_validate_best = err_validate
            update_best = update_ind
        update_ind += 1

    print 'best update: ', update_best+1
    print 'updating nodes...'
    i=0
    if len(Updates) > 0:
        while i<=update_best:
            update = Updates[i-1]
            ind, lr_update = update
            node = Layer[ind]
            node['lr'] += lr_update
            i += 1


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


def BuildLayer(NumNodes, X_train, Y_train, X_validate, Y_validate, n_iter,
               alpha, epsilon=0.01, NodeCorrection=True, BoostDecay=False,
               UltraBoosting=False, g_tol=0.01, g_final=0.0000001,
               threshold=-0.01, minibatch=False, nodeCV_size=0.1):
    Layer = InitLayer(X_train, Y_train, X_validate, Y_validate, n_iter, alpha,
                      epsilon=epsilon, minibatch=minibatch,
                      nodeCV_size=nodeCV_size)
    i = 0
    BadCount = 0
    while i < NumNodes:
        if NodeCorrection:  # check if nodes should be corrected/boosted
            #BoostNodes(Layer=Layer, X_train=X_validate, Y_train=Y_validate,
            #           epsilon=epsilon, g_tol=g_tol, threshold=threshold)
            print 'generating boosting path..'
            ValidatedNodes(Layer=Layer, X_train=X_train, Y_train=Y_train,
                           X_validate=X_validate, Y_validate=Y_validate,
                           epsilon=epsilon, g_tol=g_tol, threshold=threshold)
            [_, lam, ind], Y_pseudo, _ = CheckLayer(Layer, X_validate, Y_validate,
                                                    threshold=threshold)
            Node = NewNode(Layer, X_train, Y_train, X_validate, Y_validate,
                           n_iter=n_iter, alpha=alpha, epsilon=epsilon,
                           minibatch=minibatch, nodeCV_size=nodeCV_size)
            if EvalNode(Node, X_validate, Y_pseudo) > lam:  # > best node?
                print "adding node number ", i+2
                UsefulNode = AddNode(Layer, Node, X_validate, Y_validate)
                print "New Node is Useful: ", UsefulNode
                if UsefulNode:
                    BadCount = 0
                    i += 1
                    if BoostDecay:
                        epsilon = epsilon * i / (i+1)
                else:
                    BadCount += 1
                    if BadCount > 5:
                        break

            else:
                print "New node after boosting is not good enough!"
                g_tol = g_tol / 2.0
                if g_tol < g_final:
                    break
                print "reducing g_tol to: ", g_tol
    if NodeCorrection and UltraBoosting:
        print "starting UltraBoosting..."
        while g_tol > g_final:
            BoostNodes(Layer=Layer, X_train=X_validate, Y_train=Y_validate,
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
                    boostCV_size=0.2, nodeCV_size=0.1, NodeCorrection=True,
                    BoostDecay=False, UltraBoosting=False, g_final=0.0000001,
                    g_tol=0.01, threshold=-0.01, minibatch=False,
                    SymmetricLabels=False):

    print "creating training, validation, and testing sets..."
    train_test = train_test_split(X, Y, test_size=test_size)
    x_train, X_test, y_train, Y_test = train_test
    train_validate = train_test_split(x_train, y_train, test_size=boostCV_size)
    X_train, X_validate, Y_train, Y_validate = train_validate

    print 'fitting scalers...tranforming data...'
    if SymmetricLabels:
        X_train, X_train_inds = FoldLabels(X_train)
        X_validate, X_validate_inds = FoldLabels(X_validate)
        X_test, X_test_inds = FoldLabels(X_test)
    X_train, X_train_scaler = Preprocess(X_train)
    X_validate, X_validate_scaler = Preprocess(X_validate)
    X_test, X_test_scaler = Preprocess(X_test)
    Y_train, Y_train_scaler = Preprocess(Y_train)
    Y_validate, Y_validate_scaler = Preprocess(Y_validate)
    Y_test, Y_test_scaler = Preprocess(Y_test)

    print "initializing layer.."
    print "building layer..."
    start = timeit.default_timer()
    Layer = BuildLayer(NumNodes=NumNodes-1, X_train=X_train, Y_train=Y_train,
                       X_validate=X_validate, Y_validate=Y_validate,
                       minibatch=minibatch, n_iter=n_iter, alpha=alpha,
                       epsilon=epsilon, threshold=threshold,
                       NodeCorrection=NodeCorrection, BoostDecay=BoostDecay,
                       UltraBoosting=UltraBoosting)
    stop = timeit.default_timer()

    print "Layer Building RunTime: ", stop - start
    print "number of nodes in layer: ", len(Layer.keys())

    pred_train = 0
    pred_validate = 0
    pred_test = 0
    for ind in Layer.keys():
        node = Layer[ind]
        predict = node['predict']
        pred_train += predict(X_train) * node['lr']
        pred_validate += predict(X_validate) * node['lr']
        pred_test += predict(X_test) * node['lr']

    #plt.scatter(X_train, Y_train)
    #plt.scatter(X_train, pred_train)
    #plt.show()
    #sys.exit()

    print "Running Adabost with LR for comparison..."
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import AdaBoostRegressor
    clf = AdaBoostRegressor(loss='square', n_estimators=NumNodes)
    clf.fit(X_train, Y_train)
    pred_clf = clf.predict(X_test)

    print "Final layer results:"

    Y_train = Postprocess(Y_train, Y_train_scaler)
    pred_train = Postprocess(pred_train, Y_train_scaler)
    err_train = numpy.mean(abs(Y_train - pred_train)**2)

    print "Prediction on train data: ", pred_train
    print "actual train data: ", Y_train
    print "train error: ", err_train

    Y_validate = Postprocess(Y_validate, Y_validate_scaler)
    pred_validate = Postprocess(pred_validate, Y_validate_scaler)
    err_validate = numpy.mean(abs(Y_validate - pred_validate)**2)

    print "Prediction on validation data: ", pred_validate
    print "actual validation data: ", Y_validate
    print "validation error: ", err_validate

    Y_test = Postprocess(Y_test, Y_test_scaler)
    pred_test = Postprocess(pred_test, Y_validate_scaler)
    err_test = numpy.mean(abs(Y_test - pred_test)**2)

    print "Prediction on test data: ", pred_test
    print "actual test data: ", Y_test
    print "test error: ", err_test

    pred_clf_t = Postprocess(pred_clf, Y_test_scaler)

    err_AB_transformed = numpy.mean(abs(pred_clf_t - Y_test)**2)
    print "Scikit's Adaboost with LR on transformed data, test error: ",
    print err_AB_transformed

    X_train = Postprocess(X_train, X_train_scaler)
    X_test = Postprocess(X_test, X_test_scaler)

    clf = AdaBoostRegressor(loss='square', n_estimators=NumNodes)
    clf.fit(X_train, Y_train)
    pred_clf_raw = clf.predict(X_test)
    err_AB_raw = numpy.mean(abs(pred_clf_raw - Y_test)**2)
    print "Scikit's Adaboost with LR on original data, test error: ",
    print err_AB_raw

    if SymmetricLabels:
        X_train = unFoldLabels(X_train, X_train_inds)
        X_validate = unFoldLabels(X_validate, X_validate_inds)
        X_test = unFoldLabels(X_test, X_test_inds)

    errs = [err_train, err_validate, err_test, err_AB_raw, err_AB_transformed]
    results = [X_test, Y_test, pred_test]

    return [errs, results]
