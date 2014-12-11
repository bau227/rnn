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
import rnn_baseline

###########################  Run All Baseline Tests ##########################

STOCK_LIST = ['MSFT', 'GOOG', 'CSCO', 'IBM','AKAM', 'ADBE', 'AMZN', 'INTC', 'ORCL', 'NVDA']

def logisticTestAll(test_day):
    if test_day == 1:
       print "Logistic Regression test for 1 day"
    elif test_day == 24:
       print "Logistic Regression test for 1 month"
    else:
       print "Logistic Regression test for %i days" % test_day
    print "----------------------------------------\n"
    totalErrorRate = 0.0
    for s in STOCK_LIST:
        stockErrorRate = 0.0
        for i in range(4, 10):
            stockErrorRate += rnn_baseline.logisticTest(s, i, test_day)
        averageRate = stockErrorRate / 6.0
        print "average test error rate on %s: %f" % (s, averageRate)
        totalErrorRate += averageRate
        print
    print "average test error rate (all): " , (totalErrorRate / float(len(STOCK_LIST)))

def svmTestAll(test_day, svm_params):
    print
    if test_day == 1:
       print "Support Vector Machines test for 1 day"
    elif test_day == 24:
       print "Support Vector Machines test for 1 month"
    else:
       print "Support Vector Machines test for %i days" % test_day
    print "----------------------------------------"
    totalErrorRate = 0.0
    for s in STOCK_LIST:
        stockErrorRate = 0.0
        for i in range(4, 10):
            stockErrorRate += rnn_baseline.svmTest(s, i, test_day, svm_params)
        averageRate = stockErrorRate / 6.0
        print "average test error rate on %s: %f" % (s, averageRate)
        totalErrorRate += averageRate
        print
    print "average test error rate (all): " , (totalErrorRate / float(len(STOCK_LIST)))
    
def runAllTests(demo_mode = False):
    
    # init DataFactory() object
    d = rnn_features.DataFactory()
    d.dataDir = "./data"
    d.quoteDir = "./quotes"
    d.trainDir = "./train"
    d.testDir = d.trainDir
    d.labelType = 'S'  
    
    #  transform all raw quotes into features adapted for Logistic Regression
    print "create dataset for logistic regression..."
    for s in STOCK_LIST:
        d.createDataSet(s)
        
    print "run logistic regression tests over all stocks..."
    #perform train on one month and tests on next month varying test days    
    for i in [1, 3, 7, 11, 24]:
        logisticTestAll(i)

    d.labelType = 'A'  # for SVM label [-1,+1] fits the algorithm 
    # now we have to re-run createDataSet() to set the new labeling
    # transform all raw quotes into features adapted for SVM
    print "create dataset for support vector machines..."
    for s in STOCK_LIST:
        d.createDataSet(s)
        
    print "run svm (linear kernel) tests oon all stocks..."    
    svmParams = [5, 0.0001, 1000, 'lin', 1.2]
    for i in [1, 3, 7, 11, 24]:
        svmTestAll(i, svmParams)
    
    print "run svm (rbf kernel) tests oon all stocks..."    
    svmParams = [5, 0.0001, 1000, 'rbf', 1.2]
    for i in [1, 3, 7, 11, 24]:
        svmTestAll(i, svmParams)
    
    if demo_mode:
        print
        print "load a sample train set (MSFT, Apr) from %s..." %d.getFileName('D', 'MSFT')    
        d.loadDataSet('MSFT', [4])
        print "save a sample train set (MSFT, Apr) as %s..." % d.getFileName('TR', 'MSFT') 
        d.saveTrainSet()
        print "load a sample train set (MSFT, May) from %s..." %d.getFileName('D', 'MSFT')     
        d.loadDataSet('MSFT', [5])
        print "save a sample train set (MSFT, May) as %s..." % d.getFileName('T', 'MSFT')
        d.saveTestSet()
        print "\n\ndone!\n"

def demo():
    runAllTests(True)

if __name__ == '__main__':
     demo()
    