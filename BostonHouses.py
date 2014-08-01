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

print "----This Script uses NNBuilder on the Boston/Diabetes dataset-----"

# import some data to play with

#iris = datasets.load_boston()
iris = datasets.load_diabetes()
X = iris.data
Y = iris.target
'''
# This code block runs the algorithm once,
# then makes 3D plot of the test data and predictions
data = RunLayerBuilder(NumNodes=20, X=X, Y=Y, n_iter=5000, alpha=0.0,
                             epsilon=1.0, test_size=0.25,  boostCV_size=0.15,
                             nodeCV_size=0.1, BoostDecay=True,
                             g_final=0.000001, g_tol=0.2, minibatch=True,
                             SymmetricLabels=False)

errs, N, results = data
X_test, Y_test, pred_clf_raw = results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test)
ax.scatter(X_test[:, 0], X_test[:, 1], pred_clf_raw, color='red')
plt.show()
sys.exit()
'''

# This code block runs the algorithm 5 times, then makes nothced boxplot of
# performance of the algorithm and the best out-of-the-box algorithms on
# scikit, including AdaBoost, LogitBoost, and linear/nonlinear SVM
NodeNum_Err = {}
err_train_list = []
err_validate_list = []
err_test_list = []
err_AB_list = []
err_LB_list = []
err_SVM_lin_list = []
err_SVM_rbf_list = []
for i in range(5):
    [errs, results,
        N] = RunLayerBuilder(NumNodes=40, X=X, Y=Y, n_iter=5000, alpha=0.0,
                             epsilon=1.0, test_size=0.25,  boostCV_size=0.15,
                             nodeCV_size=0.18, BoostDecay=True,
                             g_final=0.000001, g_tol=0.2, minibatch=True,
                             SymmetricLabels=False)

    [err_train, err_validate, err_test,
        err_AB, err_LB, err_SVM_lin, err_SVM_rbf] = errs
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
plt.title('Diabetes All features, EarlyStopping w Shuffled Validation, '
          + str(trials) + ' reps')
plt.ylabel('Avg Sqaure Error (Output Range: 25-346)')
plt.show()
