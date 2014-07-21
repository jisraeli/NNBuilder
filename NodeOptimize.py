import numpy
import theano
import sys
import theano.tensor as T
from theano import pp
from sklearn import datasets



def OptimalNode(x_train, y_train):
    rng = numpy.random

    feats = len(x_train[0, :])
    print "shape of x: ", numpy.shape(x_train)
    D = (x_train, y_train)
    training_steps = 10000

    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(rng.randn(feats), name="w")
    b = theano.shared(0., name="b")
    a = theano.shared(1., name="a")
    print "Initialize node as:"
    print w.get_value(), b.get_value(), a.get_value()
    print "w is of length: ", len(w.get_value())

    # Construct Theano expression graph
    p_1 = a / (1 + T.exp(-T.dot(x, w) - b))  # Probability that target = 1
    prediction = p_1 > 0.5                    # The prediction thresholded
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)  # Cross-entropy loss
    cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
    gw, gb, ga = T.grad(cost, [w, b, a])     # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)
    #print "w derivative: ", pp(gw), "\n"
    #print "b derivative: ", pp(gb), "\n"
    #print "a derivative: ", pp(ga)
    #sys.exit()
    # Compile
    train = theano.function(inputs=[x, y],
                            outputs=[prediction, xent],
                            updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb),
                                     (a, a - 0.1 * ga)))

    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])

    print "Optimized Node:"
    print w.get_value(), b.get_value(), a.get_value()
    print "target values for D:", D[1]
    print "prediction on D:", predict(D[0])
    print "error: ", 1.0*sum(D[1]!=predict(D[0])) / len(D[1])

    return D

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :3]  # we only take the first two features.
Y = iris.target
inds = Y<2
Y = Y[inds]
X = X[inds, :]
Node = OptimalNode(X, Y)
