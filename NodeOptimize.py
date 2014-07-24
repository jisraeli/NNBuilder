import numpy
import theano
import sys
import theano.tensor as T
from theano import pp


def OptimalNode(x_train, y_train, Regression=True, Classification=False,
                bias=False, n_iter=5, alpha=0.01):
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
    D = (x_train, y_train)
    training_steps = n_iter
    #print "training steps: ", training_steps
    #print "penalty strength: ", alpha
    #print "Uses bias: ", bias

    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(rng.randn(feats), name="w")
    b = theano.shared(rng.randn(1)[0], name="b")
    a = theano.shared(1.0, name="a")
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
        pred, err = train(D[0], D[1])

    #print "Optimized Node:"
    #print w.get_value(), b.get_value(), a.get_value()
    #print "target values for D:", D[1]
    #print "prediction on D:", predict(D[0])
    #print "error: ", 1.0 * sum(abs(D[1] - predict(D[0]))) / len(D[1])

    Node = {}
    Node['w'] = w.get_value()
    Node['b'] = b.get_value()
    Node['a'] = a.get_value()
    Node['predict'] = predict
    return Node
