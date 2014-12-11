#!/usr/bin/env python

__author__ = "Anwar Sleiman Haidar, Benjamin Au"
__copyright__ = "Copyright 2014, CS221, Project: Recurrent Neural Networks for Stocks"
__credits__ = ["ASH"]
__license__ = ""
__version__ = "0.9"
__maintainer__ = "Anwar Sleiman Haidar"
__email__ = "asleiman@stanford.edu"
__status__ = "beta"


import rnn_features
from numpy import *
from svm_lib import *


############################ Logistic Regression #####################################

# sigmoid function classifier, deviation accounts for insignificant change
def classifySigmoid(v, weights):
    probability = getSigmoid(sum(v * weights))
    if probability > 0.5: 
        return 1.0
    else:
        return 0.0
    
def getSigmoid(v):
    return 1.0/(1 + exp(-v))
    
# stochastic gradient ascent (constant eta)
def SGA(data, label, eta = 0.001, num_iteration = 10, train_only = False):
    m,n = shape(data)
    weights = ones(n)          #initialize to all ones
    for j in range(num_iteration):
        for i in range(m):
            h = getSigmoid(sum(data[i]*weights))
            error = label[i] - h
            weights = weights + eta * error * data[i]
        if train_only:
            print 'training error ', abs(error)
    return weights

# assume data already created by rnn_features.DataFactory().createDataSet()
def logisticTest(ticker_symbol, train_month, test_day, train_only = False):
    # test on different days up to a month
    if test_day not in [1, 3, 7, 11, 24]:
         print "test_day value must be one of [1, 3, 7, 11, 24]"
         return 0
    if int(train_month) < 1 or int(train_month) > 12:
         print "train_month value must be between 1 and 12"
         return 0
    d = rnn_features.DataFactory()
    labelList, featureList = d.loadDataSet(ticker_symbol, [train_month])
    # training occurs here    
    trainWeights = SGA(array(featureList), labelList, train_only)  
    # load next month data for testing
    labelList, featureList = d.loadDataSet(ticker_symbol, [train_month + 1])    
    errorCount = 0 
    dataCount = 0.0
    for i in range(len(labelList)):
        if i >= test_day: break
        dataCount += 1.0
        predict = int(classifySigmoid(array(featureList[i]), trainWeights))
        if predict != int(labelList[i]):
            errorCount += 1
    if dataCount != 0:
        errorRate = (float(errorCount)/dataCount)
        print "test error rate on %s is: %f" % (ticker_symbol, errorRate)
    return errorRate

############################## Support Vector Machines #################################


def svmTest (ticker_symbol, train_month, test_day, params, train_only = False):
    # tests on different days up to a month
    if test_day not in [1, 3, 7, 11, 24]:
         print "test_day value must be one of [1, 3, 7, 11, 24]"
         return 0
    if int(train_month) < 1 or int(train_month) > 12:
         print "train_month value must be between 1 and 12"
         return 0
    d = rnn_features.DataFactory()
    labelList, featureList = d.loadDataSet(ticker_symbol, [train_month])
    # training starts here         
    # params [C, tolerance, Iteration, 'rbf' or 'lin', Sigma]
    b, alphas = smoP(featureList, labelList,params[0],params[1],params[2],\
                    (params[3],params[4]))
    dataArray = mat(featureList)
    labelArray = mat(labelList).transpose()
    svIdx = nonzero(alphas.A > 0)[0]
    svVec = dataArray[svIdx]
    svLab = labelArray[svIdx]
    #print "found %d support vectors" % shape(sv_vec)[0]
    m, n = shape(dataArray)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svVec, dataArray[i,:],(params[3], params[4]))
        predict = kernelEval.T * multiply(svLab, alphas[svIdx]) + b

        if sign(predict) != sign(labelList[i]):
            errorCount += 1
    # print "training error rate on %s is: %f" % (ticker_symbol, float(errorCount)/m)
    # load next month data for testing
    labelList, featureList = d.loadDataSet(ticker_symbol, [train_month + 1])
    dataArray = mat(featureList)
    labelArray = mat(labelList)
    errorCount = 0    
    m, n = shape(dataArray)
    for i in range(m):
        kernelEval = kernelTrans(svVec, dataArray[i,:],(params[3], params[4]))
        predict = kernelEval.T * multiply(svLab, alphas[svIdx]) + b
        if sign(predict) != sign(labelList[i]):
            errorCount += 1
    errorRate = float(errorCount)/m
    print "test error rate on %s is: %f" % (ticker_symbol, errorRate)
    return errorRate