import numpy as np
import theano
import sys
import theano.tensor as T
from theano import pp
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import math
import IPython
from LayerBuilder import*
import timeit
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D

print "----This Script tests different validation strategies----"

# import some data to play with

Data = 'Diabetes'
if Data == 'Bostom':
    iris = datasets.load_boston()
elif Data == 'Diabetes':
    iris = datasets.load_diabetes()
else:
    print 'What is this dataset? exiting.'
    sys.exit()
print 'Using ', Data, 'dataset...'
X = iris.data
Y = iris.target

err_train_shuffled_list = []
err_validate_shuffled_list = []
err_test_shuffled_list = []
err_train_uniform_list = []
err_validate_uniform_list = []
err_test_uniform_list = []
for i in range(20):
    [errs_shuffled, results,
        N] = RunLayerBuilder(NumNodes=40, X=X, Y=Y, n_iter=2000, alpha=0.0,
                             epsilon=1.0, test_size=0.25,  boostCV_size=0.15,
                             nodeCV_size=0.18, Validation='Uniform',
                             minibatch=False, SymmetricLabels=False,
                             TypeList=['Sigmoid'])
    [err_train_shuffled, err_validate_shuffled,
        err_test_shuffled]= errs_shuffled
    err_train_shuffled_list.append(err_train_shuffled)
    err_validate_shuffled_list.append(err_validate_shuffled)
    err_test_shuffled_list.append(err_test_shuffled)

    [errs_uniform, results,
        N] = RunLayerBuilder(NumNodes=40, X=X, Y=Y, n_iter=2000, alpha=0.0,
                             epsilon=1.0, test_size=0.25,  boostCV_size=0.15,
                             nodeCV_size=0.18, Validation='Uniform',
                             minibatch=False, SymmetricLabels=False,
                             TypeList=['Sigmoid', 'Gaussian'])
    [err_train_uniform, err_validate_uniform,
        err_test_uniform]= errs_uniform
    err_train_uniform_list.append(err_train_uniform)
    err_validate_uniform_list.append(err_validate_uniform)
    err_test_uniform_list.append(err_test_uniform)

trials = len(err_train_shuffled_list)
err_train_shuffled_list = np.asarray(err_train_shuffled_list)
err_validate_shuffled_list = np.asarray(err_validate_shuffled_list)
err_test_shuffled_list = np.asarray(err_test_shuffled_list)

print 'plotting results...'
fig, ax1 = plt.subplots(figsize=(10, 6))
data = [err_train_shuffled_list, err_validate_shuffled_list,
        err_test_shuffled_list,
        err_train_uniform_list, err_validate_uniform_list,
        err_test_uniform_list]
dataNames = ['training_Sigmoid', 'validation_Sigmoid', 'testing_Sigmoid',
             'training_S+G', 'validation_S+G', 'testing_S+G']
bp = plt.boxplot(data, notch=1, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
xtickNames = plt.setp(ax1, xticklabels=np.repeat(dataNames, 1))
plt.setp(xtickNames, rotation=15, fontsize=8)
plt.title('Diabetes, Unif Valid., Sigmoid+varGaussian+, '
          + str(trials) + ' reps')
plt.ylabel('Avg Sqaure Error (Output Range: 25-346)')
plt.show()
