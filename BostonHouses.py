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

print "----This Script uses NNBuilder on the Boston house-prices dataset-----"

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


print "initializing layer.."
K = 7
print "building layer..."
Layer = BuildLayer(NumNodes=K, X_train=X_train, Y_train=Y_train, n_iter=20000,
                   alpha=0.15, epsilon=0.1, NodeCorrection=True,
                   BoostDecay=True, UltraBoosting=True, threshold=-0.0002)
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

Y_train = Postprocess(Y_train, Y_train_scaler)
pred_train = Postprocess(pred_train, Y_train_scaler)

print "Prediction on train data: ", pred_train
print "actual train data: ", Y_train
print "train error: ", 1.0 * sum(abs(Y_train-pred_train)**2) / len(Y_train)

Y_test = Postprocess(Y_test, Y_test_scaler)
pred_test = Postprocess(pred_test, Y_test_scaler)

print "Prediction on test data: ", pred_test
print "actual test data: ", Y_test
print "test error: ", 1.0 * sum(abs(Y_test-pred_test)**2) / len(Y_test)

pred_clf = Postprocess(pred_clf, Y_test_scaler)

err = numpy.mean((pred_clf - Y_test)**2)
print "Scikit's Adaboost with LR on transformed data, test error: ", err

X_train = Postprocess(X_train, X_train_scaler)
X_test = Postprocess(X_test, X_test_scaler)

clf = AdaBoostRegressor(base_estimator=LogisticRegression(), n_estimators=K+1,
                        loss='square')
clf.fit(X_train, Y_train)
pred_clf = clf.predict(X_test)
err = numpy.mean((pred_clf - Y_test)**2)
print "Scikit's Adaboost with LR on original data, test error: ", err
