from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import theano
import theano.tensor as T
from NodeOptimize import OptimalNode


'''
Construct 1 hidden layer NN using boosting:
    initialize k=0, Fi=0
    loop {
        compute "pseudo response" l = -dL(Y, F) / dF (square loss: Yi-Fi(v))
        (b, zeta) = Minimize [l - zeta*S(x; b)]
        eta = Minimize L[Y, F+eta*S(x;b)]
        delta_v=epsilon*eta (double check this is right!)
        k = k + 1
        b_k = b (from optimization w.r.t "pseudo" response)
        a_k = delta_v
        F = F +  delta_v*S(x;b_k)
    }
'''


def Init_Layer(X_transformed, y):
    '''
    X_transformed: output from the most recent layer
    y: actual labels

    return: new layer with a single node
    '''

    Layer = {}
    clf = OptimalNode(X_transformed, y)
    Layer['1'] = clf

    return Layer


def Compute_Layer(Layer, X_transformed):
    '''
    X_transformed: output from the most recent layer

    return: Layer's predictions
    '''
    y_predict = 0
    predict = theano.function(inputs=[x], outputs=prediction)
    for node in Layer:
        y_predict += predict(node[0])

    return y_predict


def Extend_Layer(Layer, X_transformed, y, index):
    '''
    X_transformed: output from the most recent layer
    y: actual labels

    return: Layer extended with a new node
    '''
    l = y - Compute_Layer(Layer, X_transformed)
    node = OptimalNode(X_transformed, l)
    Layer[index] = node




