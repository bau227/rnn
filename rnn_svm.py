from rnn_util import *
from svm_lib import *

K1 = 1.2

def svm_test (ticker_stock, directory = DEFAULT_TRAIN_TEST_DIR, ext_train = EXT_TRAIN, ext_test = EXT_TEST):

    f_train = open(directory + '/' + ticker_stock + ext_train)
    train_data = [] 
    train_label = []
    for line in f_train.readlines():
        c = line.strip().split()
        l =[]
        for i in range(1, NUM_FEATURES):
            l.append(float(c[i]))
        train_data.append(l)
        train_label.append(int(c[0]))
    b, alphas = smoP(train_data, train_label, 5, 0.0001, 1000, ('rbf', K1))

    data_array = mat(train_data)
    label_array = mat(train_label).transpose()
    sv_idx = nonzero(alphas.A > 0)[0]
    sv_vec = data_array[sv_idx]
    sv_lab = label_array[sv_idx]
    
    #print "found %d support vectors" % shape(sv_vec)[0]
    m, n = shape(data_array)
    error_count = 0
    for i in range(m):
        kernel_eval = kernelTrans(sv_vec, data_array[i,:],('rbf', K1))
        prediction = kernel_eval.T * multiply(sv_lab, alphas[sv_idx]) + b

        if sign(prediction) != sign(train_label[i]):
            error_count += 1
    #print "training error rate on %s  is: %f" % (ticker_stock, float(error_count)/m)
    
    f_test = open(directory + '/' + ticker_stock + ext_test)
    test_data = []
    test_label =[]
    for line in f_test.readlines():
        c = line.strip().split()
        l =[]
        for i in range(1, NUM_FEATURES):
            l.append(float(c[i]))
        test_data.append(l)
        test_label.append(int(c[0]))    
    data_array = mat(test_data)
    label_array = mat(test_label)
    error_count = 0    
    m, n = shape(data_array)
    for i in range(m):
        kernel_eval = kernelTrans(sv_vec, data_array[i,:],('rbf', K1))
        prediction = kernel_eval.T * multiply(sv_lab, alphas[sv_idx]) + b
        if sign(prediction) != sign(test_label[i]):
            error_count += 1
    print "test error rate on %s is: %f" % (ticker_stock, float(error_count)/m)
    
    return float(error_count)/m

def svm_multimonth_test(ticker_stock, same_month = False, directory_data = DEFAULT_DATA_DIR,\
                             directory_train_test = DEFAULT_TRAIN_TEST_DIR,\
                             ext_data = EXT_DATA, ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    Test all entries/month for one stock ticker
    calls createtest() for each month and hands it over to logistic_test() for the grunt work
    If same_month is True calls create_traintest2() instead of create_traintest()
    """
    total_error = 0.0
    num_test = 0.0
    for n in range(4, 10):
        num_test += 1
        if same_month:
            create_traintest2(ticker_stock, n, directory_data, directory_train_test, ext_data, ext_train, ext_test)
        else:
            create_traintest(ticker_stock, n, directory_data, directory_train_test, ext_data, ext_train, ext_test)
        total_error += svm_test(ticker_stock, directory_train_test, ext_train, ext_test)
    print 'average error rate (over %d tests--one/month) for %s is: %f' % ( num_test, ticker_stock, total_error/float(num_test))
    return total_error/float(num_test)

def svm_multimonth_test_all(ticker_list = STOCK_LIST, same_month = False, directory_data = DEFAULT_DATA_DIR,\
                                directory_train_test = DEFAULT_TRAIN_TEST_DIR, ext_data = EXT_DATA, \
                                ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    Test all entries/month for all stocks in TICKER_LIST (default)
    calls logistic_multimomnth_test() iteratively over all stock symbols
    """
    total_error = 0.0
    num_test = 0.0
    for t in ticker_list:
       num_test += 1
       total_error += svm_multimonth_test(t, same_month, directory_data, directory_train_test, ext_data, ext_train, ext_test)
    print 'average error rate across all tests (%d stocks) is: %f' % (num_test, total_error/float(num_test))  