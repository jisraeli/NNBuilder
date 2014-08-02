import numpy
import theano
import sys
import theano.tensor as T
from theano import pp
from sklearn.cross_validation import train_test_split
import IPython


def OptimalNode(x_train, y_train, Regression=True, Classification=False,
                bias=False, n_iter=5, alpha=0.01, minibatch=False):
    '''
    inputs
        x_train: training features
        y_train: response variable
        n_iter: # of iterations for SGD
        alpha: strength of L2 penalty (default penalty for now)
    outputs
        Node: dictionary with Node parameters an predict method
    '''

    rng = numpy.random

    feats = len(x_train[0, :])
    D = [x_train, y_train]
    training_steps = n_iter
    #print "training steps: ", training_steps
    #print "penalty strength: ", alpha
    #print "Uses bias: ", bias

    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(rng.uniform(low=-0.25, high=0.25, size=feats), name="w")
    b = theano.shared(rng.randn(1)[0], name="b")
    a = theano.shared(abs(rng.randn(1)[0]), name="a")
    #print "Initialize node as:"
    #print w.get_value(), b.get_value(), a.get_value()

    # Construct Theano expression graph
    if bias:
        p_1 = -0.5 + a / (1 + T.exp(-T.dot(x, w) - b))
    else:
        p_1 = a / (1 + T.exp(-T.dot(x, w)))
    prediction = p_1 > 0.5
    if Classification:
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)  # Cross-entropy loss
    elif Regression:
        xent = 0.5 * (y - p_1)**2
    if alpha == 0:
        cost = xent.mean()  # The cost to minimize
    else:
        cost = xent.mean() + alpha * ((w ** 2).sum())
    if bias:
        gw, gb, ga = T.grad(cost, [w, b, a])
    else:
        gw, ga = T.grad(cost, [w, a])  # Compute the gradient of the cost

    # Compile
    Node = {}
    Node['Path'] = {}
    NodePath = Node['Path']
    if bias:
        train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                                updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb),
                                         (a, a - 0.1 * ga)))
    else:
        train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                                updates=((w, w - 0.1 * gw), (a, a - 0.1 * ga)))

    predict = theano.function(inputs=[x], outputs=p_1)

    # Train
    for i in range(training_steps):
        if minibatch:
            batch_split = train_test_split(x_train, y_train, test_size=0.2)
            _, D[0], _, D[1] = batch_split
            pred, err = train(D[0], D[1])
        elif not minibatch:
            pred, err = train(D[0], D[1])
        NodePath[str(i)] = {}
        NodePath[str(i)]['w'] = w.get_value()
        NodePath[str(i)]['b'] = b.get_value()
        NodePath[str(i)]['a'] = a.get_value()

    Node['w'] = w.get_value()
    Node['b'] = b.get_value()
    Node['a'] = a.get_value()
    Node['predict'] = predict

    return Node


def OptimalGaussian(x_train, y_train, Regression=True, Classification=False,
                    bias=False, n_iter=5, alpha=0.01, minibatch=False):
    '''
    inputs
        x_train: training features
        y_train: response variable
        n_iter: # of iterations for SGD
        alpha: strength of L2 penalty (default penalty for now)
    outputs
        Gaussian Node: dictionary with Node parameters an predict method
    '''

    rng = numpy.random

    feats = len(x_train[0, :])
    D = [x_train, y_train]
    training_steps = n_iter
    #print "training steps: ", training_steps
    #print "penalty strength: ", alpha
    #print "Uses bias: ", bias

    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(rng.uniform(low=-0.25, high=0.25, size=feats), name="w")
    b = theano.shared(abs(rng.randn(1)[0]), name="b")
    a = theano.shared(abs(rng.randn(1)[0]), name="a")
    #print "Initialize node as:"
    #print w.get_value(), b.get_value(), a.get_value()

    # Construct Theano expression graph
    if bias:
        p_1 = a * T.exp(-0.5 / (b**2) * T.dot((x - w).T, (x - w)))
    else:
        p_1 = a * T.exp(-0.5 / (1**2) * T.dot((x - w).T, (x - w)))
    prediction = p_1 > 0.5
    if Regression:
        xent = 0.5 * (y - p_1)**2
    if alpha == 0:
        cost = xent.mean()  # The cost to minimize
    else:
        cost = xent.mean() + alpha * ((w ** 2).sum())
    if bias:
        gw, gb, ga = T.grad(cost, [w, b, a])
    else:
        gw, ga = T.grad(cost, [w, a])  # Compute the gradient of the cost

    # Compile
    Node = {}
    Node['Path'] = {}
    NodePath = Node['Path']
    if bias:
        train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                                updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb),
                                         (a, a - 0.1 * ga)))
    else:
        train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                                updates=((w, w - 0.1 * gw), (a, a - 0.1 * ga)))

    predict = theano.function(inputs=[x], outputs=p_1)

    # Train
    for i in range(training_steps):
        if minibatch:
            batch_split = train_test_split(x_train, y_train, test_size=0.2)
            _, D[0], _, D[1] = batch_split
            pred, err = train(D[0], D[1])
        elif not minibatch:
            pred, err = train(D[0], D[1])
        NodePath[str(i)] = {}
        NodePath[str(i)]['w'] = w.get_value()
        NodePath[str(i)]['b'] = b.get_value()
        NodePath[str(i)]['a'] = a.get_value()

    Node['w'] = w.get_value()
    Node['b'] = b.get_value()
    Node['a'] = a.get_value()
    Node['predict'] = predict

    return Node





def EarlyStopNode(Node, x_validate, y_validate):
    '''
    Creates validation set
    Evaluates Node's path on validation set
    Chooses optimal w in Node's path based on validation set
    '''
    x = T.matrix("x")
    y = T.vector("y")
    w = T.vector("w")
    b = T.dscalar("b")
    a = T.dscalar("a")
    p_1 = -0.5 + a / (1 + T.exp(-T.dot(x, w) - b))
    xent = 0.5 * (y - p_1)**2
    cost = xent.mean()
    loss = theano.function(inputs=[x, y, w, b, a], outputs=cost)

    Path = Node['Path'].keys()
    Path = map(int, Path)
    Path.sort()
    best_node = {}
    best_node_ind = 0
    best_loss = numpy.mean(y_validate**2)
    losses = []
    for ind in Path:
        node = Node['Path'][str(ind)]
        l = loss(x_validate, y_validate, node['w'], node['b'], node['a'])
        losses.append(l)
        if l < best_loss:
            best_node = node
            best_node_ind = ind
            best_loss = l
    #print "path losses: ", losses
    #print "best path index: ", best_node_ind
    #print "best loss: ", best_loss
    #IPython.embed()

    Node['w'] = best_node['w']
    Node['b'] = best_node['b']
    Node['a'] = best_node['a']

    return Node
