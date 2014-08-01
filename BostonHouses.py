import numpy as np
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
import timeit
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D

print "----This Script uses NNBuilder on the Boston house-prices dataset-----"

# import some data to play with

iris = datasets.load_boston()
#X = iris.data[:, [10, 12]]
X = iris.data
Y = iris.target
#X = np.random.uniform(-1.0, 1.0, [1000, 1])
#Y = np.sin(X)
#Y = X**2
#Y = np.reshape(Y, [1000])
'''
data = RunLayerBuilder(NumNodes=10, X=X, Y=Y, n_iter=1320, alpha=0.0,
                             epsilon=1.0, test_size=0.25,  boostCV_size=0.15,
                             nodeCV_size=0.1, BoostDecay=True,
                             g_final=0.000001, g_tol=0.2, minibatch=True,
                             SymmetricLabels=False)

errs, results = data
X_test, Y_test, pred_clf_raw = results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test)
ax.scatter(X_test[:, 0], X_test[:, 1], pred_clf_raw, color='red')
plt.show()
sys.exit()
'''

NodeNum_Err = {}
err_train_list = []
err_validate_list = []
err_test_list = []
err_AB_list = []
err_LB_list = []
err_SVM_lin_list = []
err_SVM_rbf_list = []
for i in range(5):
    result, N = RunLayerBuilder(NumNodes=10, X=X, Y=Y, n_iter=5000, alpha=0.0,
                             epsilon=1.0, test_size=0.25,  boostCV_size=0.15,
                             nodeCV_size=0.18, BoostDecay=True,
                             g_final=0.000001, g_tol=0.2, minibatch=True,
                             SymmetricLabels=False)

    [err_train, err_validate, err_test,
        err_AB, err_LB, err_SVM_lin, err_SVM_rbf] = result
    err_train_list.append(err_train)
    err_validate_list.append(err_validate)
    err_test_list.append(err_test)
    err_AB_list.append(err_AB)
    err_LB_list.append(err_LB)
    err_SVM_lin_list.append(err_SVM_lin)
    err_SVM_rbf_list.append(err_SVM_rbf)

    if str(N) in NodeNum_Err.keys():
        NodeNum_Err[str(N)]['count'] += 1.0
        NodeNum_Err[str(N)]['TotalErr'] += err_test
    else:
        NodeNum_Err[str(N)] = {}
        NodeNum_Err[str(N)]['count'] = 1.0
        NodeNum_Err[str(N)]['TotalErr'] = err_test

print '#ofNodes #ofModels AvgTestError'
keys = NodeNum_Err.keys()
keys.sort()
for NumNodes in keys:
    count = NodeNum_Err[NumNodes]['count']
    TotalErr = NodeNum_Err[NumNodes]['TotalErr']
    print NumNodes, count, TotalErr / count

trials = len(err_train_list)
err_train_list = np.asarray(err_train_list)
err_validate_list = np.asarray(err_validate_list)
err_test_list = np.asarray(err_test_list)
err_AB_list = np.asarray(err_AB_list)
err_LB_list = np.asarray(err_LB_list)
err_SVM_lin_list = np.asarray(err_SVM_lin_list)
err_SVM_rbf_list = np.asarray(err_SVM_rbf_list)

print 'plotting results...'
fig, ax1 = plt.subplots(figsize=(10, 6))
data = [err_train_list, err_validate_list, err_test_list,
        err_AB_list,  err_LB_list, err_SVM_lin_list, err_SVM_rbf_list]
dataNames = ['training', 'validation', 'testing', 'AdaBoost', 'LogitBoost',
             'Linear SVM', 'Gaussian SVM']
bp = plt.boxplot(data, notch=1, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
xtickNames = plt.setp(ax1, xticklabels=np.repeat(dataNames, 1))
plt.setp(xtickNames, rotation=15, fontsize=8)
plt.title('Boston Dataset, All features, No EarlyStopping, '
          + str(trials) + ' attempts')
plt.show()


'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print 'fitting scalers...tranforming data...'
X_train, X_train_scaler = Preprocess(X_train)
X_test, X_test_scaler = Preprocess(X_test)
Y_train, Y_train_scaler = Preprocess(Y_train)
Y_test, Y_test_scaler = Preprocess(Y_test)


print "initializing layer.."
K = 7
print "building layer..."
start = timeit.default_timer()
Layer = BuildLayer(NumNodes=K, X_train=X_train, Y_train=Y_train, n_iter=5000,
                   alpha=0.15, epsilon=0.02, NodeCorrection=True,
                   BoostDecay=True, UltraBoosting=True, threshold=-0.0002,
                   minibatch=False)
stop = timeit.default_timer()

print "Layer Building RunTime: ", stop - start
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
print "train error: ", numpy.mean(abs(Y_train-pred_train)**2)

Y_test = Postprocess(Y_test, Y_test_scaler)
pred_test = Postprocess(pred_test, Y_test_scaler)

print "Prediction on test data: ", pred_test
print "actual test data: ", Y_test
print "test error: ", numpy.mean(abs(Y_test-pred_test)**2)

pred_clf = Postprocess(pred_clf, Y_test_scaler)

err = numpy.mean(abs(pred_clf - Y_test)**2)
print "Scikit's Adaboost with LR on transformed data, test error: ", err

X_train = Postprocess(X_train, X_train_scaler)
X_test = Postprocess(X_test, X_test_scaler)

clf = AdaBoostRegressor(base_estimator=LogisticRegression(), n_estimators=K+1,
                        loss='square')
clf.fit(X_train, Y_train)
pred_clf = clf.predict(X_test)
err = numpy.mean(abs(pred_clf - Y_test)**2)
print "Scikit's Adaboost with LR on original data, test error: ", err
'''
