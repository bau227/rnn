#!/usr/bin/env python

__author__ = "Anwar Sleiman Haidar, Benjamin Au"
__copyright__ = "Copyright 2014, CS221, Project: Recurrent Neural Networks for Stock Price Movement Prediction"
__credits__ = ["Peter Harrington"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Anwar Sleiman Haidar"
__email__ = "asleiman@stanford.edu"
__status__ = "beta"


"""
                   Function List

Main:

create_dataset()
create_traintest()

logistic_test()
logistic_multimonth_test()
logistic_multimonth_test_all()

prep_raw_data()
write_prep_data()

Helpers:

SGA()
get_label()
get_month()
get_change()
get_average()
get_dayofweek()
classify_sigmoid()
get_sigmoid()

Independent (not used for now):

get_EMA()
get_SMA()

"""

from numpy import *
import time
import datetime 

# Global Variables

TICKER_LIST = ['GOOG', 'MSFT', 'CSCO', 'IBM', 'AKAM', 'AAPL', 'ADBE', 'AMZN', 'INTC', 'ORCL', 'NVDA']
TICKER_MARKET = '^COMP'
TICKER_ECONOMIC = '^W5000'
EXT_DATA = '.data'
EXT_TRAIN = '.train'
EXT_TEST = '.test'
EXT_TEXT = '.csv'
TEXT_DELIMITER = ','
NUM_FEATURES = 6
DEFAULT_DATA_DIR ='.'
DEFAULT_QUOTE_DIR = '.'
DEFAULT_TRAIN_TEST_DIR = '.'

def create_dataset(ticker_list = TICKER_LIST, directory_data = DEFAULT_DATA_DIR, directory_quote = DEFAULT_QUOTE_DIR, \
                   ticker_market = TICKER_MARKET, ticker_economic = TICKER_ECONOMIC, ext_text = EXT_TEST, ext_data = EXT_DATA, delimiter = TEXT_DELIMITER, header=True):
    """
    @param self-explanatory, all have default values; typical call create_dataset()
    loops over TICKER_LIST and uses prep_raw_data and write_prep_data to load/adjust data quotes and save to .data file
    """
    for s in TICKER_LIST:
        
        data = prep_raw_data(s, directory_quote, ticker_market, ticker_economic, ext_text, delimiter, header)
        write_prep_data(s, data, directory_data, ext_data) 

# month must be between 4 and 9 for this data set
# creates a .train dataset for month and a .test dataset for month + 1
def create_traintest(ticker_stock, train_month, directory_data = DEFAULT_DATA_DIR, directory_train_test = DEFAULT_TRAIN_TEST_DIR, \
                     ext_data = EXT_DATA, ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    @param stock symbol, month (the rest are set to their defaults)
    creates train and test files from data based on a month (m)
    the test file contains the next month data
    """
    f_data = open(directory_data + '/' + ticker_stock + ext_data, 'rU')
    f_test = open(directory_train_test + '/' + ticker_stock + ext_test, 'w')
    f_train = open(directory_train_test + '/' + ticker_stock + ext_train, 'w')
    for line in f_data.readlines():
        s = line.split()
        if int(s[7]) == train_month:
            #l = ' '.join(str(x) for x in line) + '\n'
            f_train.write(line)
        elif int(s[7]) == train_month + 1:
            #l = ' '.join(str(x) for x in line) + '\n'
            f_test.write(line)

def prep_raw_data(ticker_stock, directory_quote = DEFAULT_QUOTE_DIR, ticker_market = TICKER_MARKET, ticker_economic = TICKER_ECONOMIC,\
                  ext_text = EXT_TEXT, delimiter= TEXT_DELIMITER, header=True):
    """
    @param ticker_stock = 'AAPL' (the rest are set to their defaults)
    
    N.B.
       ticker_market = '^COMP' (default for NASDAQ)
       ticker_economic = '^W5000' (default for Wilshire 5000)
    
    Initially we wanted a weighted average of several market indices as 
    a proxy for a global economic indicator. Finally we opted for the Wilshire
    because of its larger asset base.
     
    This function loads the data as downloaded from Yahoo as http://ichart.yahoo.com/table.csv?s=XXXX&a=3&b=1&c=2014&d=9&e=31&f=2014&g=d
    XXXX : Stock Ticker, and the rest of the query is start and end dates
    It builds a new list {new_data] which has the following entries:
    
    [label, day of the week, average price % change, high low price % change, stock volume % change, market volume % change, economic volume % change]
    FYI, first it builds the intermediate list [current_data]
    
    The label is the true value of the prediction. [+1, 0, -1] (price increase, same, decrease)
    
    P.S. The data will be 2 entries less than the original download (first and last entries will not be included)
    
    """
    # 'rU' is for universal support because lines are '\r' delimited and not '\n'
    f_stock = open(directory_quote + '/' + ticker_stock + ext_text, 'rU')
    f_market = open(directory_quote + '/' + ticker_market + ext_text, 'rU')
    f_economic = open(directory_quote + '/' + ticker_economic + ext_text, 'rU')
    #Files must have same number of entries in ascending date order
    current_data = []
    flag = header
    for line in f_stock.readlines():
        s = line.strip().split(delimiter)
        if flag:
            flag = not flag
        else:
            #Close Day, Day of Week, Average Price, High Low Variation, Volume Stock Variation, Volume Market Variation, Volume Economic Variation, month (placeholder used to create train and test sets)
            #Close Day is a temporary entry. It will be used to determine the true label value.
            current_data.append([float(s[4]), get_dayofweek(s[0]) + 1, get_average(s[1], s[4]), get_change(s[3], s[2]), float(s[5]), 0.0, 0.0, get_month(s[0])])
    flag = header
    i = 0
    for line in f_market.readlines():
        if flag:
            flag = not flag
            continue
        s = line.strip().split(delimiter)
        current_data[i][5] = float(s[4]) 
        i +=1  
    flag = header
    i = 0
    for line in f_economic.readlines():
        if flag:
            flag = not flag
            continue
        s = line.strip().split(delimiter)
        current_data[i][6] = float(s[4]) 
        i +=1    
    # get first entry, don't add to data
    previous_data = current_data[0]
    new_data = []
    for i in range(1, len(current_data)):
        c = current_data[i]
        if i < len(current_data) - 1:
           label = get_label(current_data[i][0], current_data[i+1][0])
           #print current_data[i+1][0], ' ', current_data[i][0], ' ', label
        else:
           label = 0
        # don't add last entry
        if i != len(current_data) - 1:   
            new_data.append([label, c[1], get_change(previous_data[2], c[2]), c[3], \
                        get_change(previous_data[4], c[4]), get_change(previous_data[5], c[5]), \
                        get_change(previous_data[6], c[6]), c[7]])
        previous_data = current_data[i]
    return new_data

def write_prep_data(ticker_name, data_list, directory_data = DEFAULT_DATA_DIR, ext_data = EXT_DATA):
    """
    just writes the list from prep_raw_data() to a data file
    """
    f_data = open(directory + '/' + ticker_name + ext_data, 'w')
    for i in range(len(data_list)):
        line = ' '.join(str(x) for x in data_list[i]) + '\n'
        f_data.write(line)

# Label +1 if next closing price is higher, -1 if lower, 0 if within the deviation margin
def get_label(v1, v2, deviation = 0.001):
    c = get_change(v1, v2)
    if c > deviation:
        return +1
    if c < -deviation:
        return -1
    return 0    

# Converts a date string into day of week [0-6]    
# In the calling function we add 1, so Mondays are 1 instead of 0.
def get_dayofweek(string_date, delimiter = '/'):
    s = string_date.split(delimiter)
    d = datetime.date(int(s[2]), int(s[0]), int(s[1]))
    return datetime.date.weekday(d)

def get_month(string_date, delimiter = '/'):
    s = string_date.split(delimiter)
    return int(s[0])
        
def get_change(v1, v2):
    if v2 == 0: return 0
    return  (float(v2) - float(v1)) / float(v2)

def get_average(v1, v2):
     return  (float(v2) + float(v1)) / 2.0
     
def get_sigmoid(v):
    return 1.0/(1 + exp(-v))

# SIMPLE MOVING AVERAGE
def get_SMA(dataset, step):
    w = numpy.repeat(1.0, step) /step
    sma = numpy.convolve(dataset, w, 'valid') 
    return sma
    
# Exponential Moving Average
def get_EMA(dataset, step):
    w = numpy.exp(numpy.linspace(-1., 0., step))
    w  /= w.sum()
    ema = numpy.convolve(dataset, w, mode='full')[:len(dataset)]
    ema[:step]=ema[step]
    return ema
    
#Sigmoid Function Classifier, deviation accounts for insignificant change
def classify_sigmoid(v, weights, deviation = 0.01):
    probability = get_sigmoid(sum(v * weights))
    #print probability
    #print sum(v * weights), ' ' , probability
    if probability > 0.5 + deviation: return  +1
    if probability < 0.5 - deviation: return -1
    return 0
    
#Stochastic Gradient Ascent (constant eta)
def SGA(data, label, eta = 0.01, num_iteration = 1):
    m,n = shape(data)
    weights = ones(n)   #initialize to all ones
    for j in range(num_iteration):
        for i in range(m):
            h = get_sigmoid(sum(data[i]*weights))
            error = label[i] - h
            weights = weights + eta * error * data[i]
        #print 'iteration: ', j, ' weights: ', weights
    return weights
    
def logistic_test(ticker_stock, directory = DEFAULT_TRAIN_TEST_DIR, ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    Main train/test function where SGA and classify_sigmoid are called
    First it start training on train set and then uses the computed weights on the test set.
    """
    f_train = open(directory + '/' + ticker_stock + ext_train)
    f_test = open(directory + '/' + ticker_stock + ext_test)
    train_data = [] 
    train_label = []
    for line in f_train.readlines():
        c = line.strip().split()
        l =[]
        for i in range(1, NUM_FEATURES):
            l.append(float(c[i]))
        train_data.append(l)
        train_label.append(int(c[0]))
    train_weights = SGA(array(train_data), train_label)
    error_count = 0 
    data_count = 0.0
    for line in f_test.readlines():
        data_count += 1.0
        c = line.strip().split()
        l =[]
        for i in range(1, NUM_FEATURES):
            l.append(float(c[i]))
        c_s = int(classify_sigmoid(array(l), train_weights))
        #print 'computed: ', c_s, 'expected: ', int(c[0]) 
        if c_s != int(c[0]):
            error_count += 1
    if data_count != 0:
        error_rate = (float(error_count)/data_count)
    print "error rate on " + ticker_stock + " is: %f" % error_rate
    return error_rate

def logistic_multimonth_test(ticker_stock, directory_data = DEFAULT_DATA_DIR, directory_train_test = DEFAULT_TRAIN_TEST_DIR,\
                             ext_data = EXT_DATA, ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    Test all entries/month for one stock ticker
    calls createtest() for each month and hands it over to logistic_test() for the grunt work
    """
    total_error = 0.0
    num_test = 0.0
    for n in range(4, 10):
        num_test += 1
        #print 'creating month ', n, 'training set, and month ', n + 1, 'test set...'
        create_traintest(ticker_stock, n, directory_data, directory_train_test, ext_data, ext_train, ext_test)
        total_error += logistic_test(ticker_stock, directory_train_test, ext_train, ext_test)
    print "average error rate (%d iterations) is: %f" % (num_test, total_error/float(num_test))

def logistic_multimonth_test_all(ticker_list = TICKER_LIST, directory_data = DEFAULT_DATA_DIR,\
                                directory_train_test = DEFAULT_TRAIN_TEST_DIR, ext_data = EXT_DATA, \
                                ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    Test all entries/month for all stocks in TICKER_LIST (default)
    calls logistic_multimomnth_test() iteratively over all stock symbols
    """
    for t in ticker_list:
      logistic_multimonth_test(t, directory_data, directory_train_test, ext_data, ext_train, ext_test)
        