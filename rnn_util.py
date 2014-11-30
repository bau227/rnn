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
                   
                                !!!WARNING!!!
                   
                   THERE IS MINIMUM (IF ANY) ERROR TESTS,
                   SILLY THINGS WHETHER OR NOT A FILE EXISTS...
                   ASSUMES THAT DATA HAS NO MISSING VALUES AND IS PROPERLY
                   FORMATTED AND ORDERED. SEE .CSV AND .DATA FILES FOR
                   MORE INFO...

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

STOCK_LIST = ['MSFT', 'GOOG','CSCO', 'IBM', 'AKAM', 'AAPL', 'ADBE', 'AMZN', 'INTC', 'ORCL', 'NVDA']
STOCK_LIST2 = ['MSFT', 'CSCO', 'AKAM', 'ADBE', 'AMZN', 'INTC', 'ORCL', 'NVDA']
MARKET_LIST = ['^COMP', '^DJIA', '^GSPC'] 
TICKER_ECONOMIC = '^W5000'
EXT_DATA = '.data'
EXT_TRAIN = '.train'
EXT_TEST = '.test'
EXT_TEXT = '.csv'
TEXT_DELIMITER = ','
BASE_FEATURES = 7
NUM_FEATURES = BASE_FEATURES + len(MARKET_LIST)
NUM_MONTH = NUM_FEATURES + 1
DEFAULT_DATA_DIR ='.'
DEFAULT_QUOTE_DIR = '.'
DEFAULT_TRAIN_TEST_DIR = '.'

def create_dataset(stock_list = STOCK_LIST, directory_data = DEFAULT_DATA_DIR, directory_quote = DEFAULT_QUOTE_DIR, \
                   market_list = MARKET_LIST, ticker_economic = TICKER_ECONOMIC, ext_text = EXT_TEXT,\
                   ext_data = EXT_DATA, delimiter = TEXT_DELIMITER, header=True):
    """
    @param self-explanatory, all have default values; typical call create_dataset()
    loops over STOCK_LIST and uses prep_raw_data and write_prep_data to load/adjust data quotes and save to .data file
    """
    for s in stock_list:
        data = prep_raw_data(s, directory_quote, market_list, ticker_economic, ext_text, delimiter, header)
        write_prep_data(s, data, directory_data, ext_data) 

# month must be between 4 and 9 for this data set
# creates a .train dataset for month and a .test dataset for month + 1
def create_traintest(ticker_stock, train_month, directory_data = DEFAULT_DATA_DIR,\
                     directory_train_test = DEFAULT_TRAIN_TEST_DIR, \
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
        if int(s[NUM_MONTH-1]) == train_month:
            f_train.write(line)
        elif int(s[NUM_MONTH-1]) == train_month + 1:
            f_test.write(line)

#same as above but split the month into 2 roughly parts 2/3, 1/3. train on the first part and test on the second.
def create_traintest2(ticker_stock, train_month, directory_data = DEFAULT_DATA_DIR,\
                     directory_train_test = DEFAULT_TRAIN_TEST_DIR, \
                     ext_data = EXT_DATA, ext_train = EXT_TRAIN, ext_test = EXT_TEST):
    """
    @param stock symbol, month (the rest are set to their defaults)
    creates train and test files from data based on a month (m)
    the test file contains the next month data
    """
    f_data = open(directory_data + '/' + ticker_stock + ext_data, 'rU')
    f_test = open(directory_train_test + '/' + ticker_stock + ext_test, 'w')
    f_train = open(directory_train_test + '/' + ticker_stock + ext_train, 'w')
    i = 0
    for line in f_data.readlines():
        s = line.split()
        if int(s[NUM_MONTH-1]) == train_month: 
            if i < 15:
                f_train.write(line)
            else:
                f_test.write(line)
            i += 1
        
def prep_raw_data(ticker_stock, directory_quote = DEFAULT_QUOTE_DIR, market_list = MARKET_LIST,\
                  ticker_economic = TICKER_ECONOMIC,\
                  ext_text = EXT_TEXT, delimiter= TEXT_DELIMITER, header=True):
    """
    @param ticker_stock = 'AAPL' (the rest are set to their defaults)
    
    N.B.
       ticker_market = '^COMP' (default for NASDAQ)
       ticker_economic = '^W5000' (default for Wilshire 5000)
    
    Initially we wanted a weighted average of several market indices as 
    a proxy for a global economic indicator. Finally we opted for the Wilshire
    because of its larger asset base.
     
    This function loads the data as downloaded from Yahoo as 
    http://ichart.yahoo.com/table.csv?s=XXXX&a=3&b=1&c=2014&d=9&e=31&f=2014&g=d
    XXXX : Stock Ticker, and the rest of the query is start and end dates
    It builds a new list {new_data] which has the following entries:
    
    [label, day of the week, average price % change, high low price % change, 
    stock volume % change, market volume % change, economic volume % change]
    FYI, first it builds the intermediate list [current_data]
    
    The label is the true value of the prediction. [+1, 0, -1] (price increase, same, decrease)
    
    P.S. The data will be 2 entries less than the original download (first and last entries will not be included)
    
    """
    # 'rU' is for universal support because lines are '\r' delimited and not '\n'
    f_stock = open(directory_quote + '/' + ticker_stock + ext_text, 'rU')
    f_economic = open(directory_quote + '/' + ticker_economic + ext_text, 'rU')
    #Files must have same number of entries in ascending date order
    current_data = []
    flag = header
    for line in f_stock.readlines():
        s = line.strip().split(delimiter)
        if flag:
            flag = not flag
        else:
            #Close Day, Day of Week, Average Price, High Low Variation, Open Price Variation, Volume Stock Variation, Volume Market Variation, Volume Economic Variation, month (placeholder used to create train and test sets)
            #Close Day is a temporary entry. It will be used to determine the true label value.
            l = []
            for i in range(NUM_MONTH):
               l.append(0)
            l[0] = float(s[4])
            #l[1] = 1.0
            l[1] = get_dayofweek(s[0]) + 1
            l[2] = get_average(s[1], s[4])
            l[3] = float(s[1])
            l[4] = get_change(s[3], s[2])
            l[5] = float(s[5])
            l[NUM_MONTH-1] = get_month(s[0])
            current_data.append(l)
            #current_data.append([float(s[4]), get_dayofweek(s[0]) + 1, get_average(s[1], s[4]), float(s[1]), get_change(s[3], s[2]), float(s[5]), 0.0, 0.0, get_month(s[0])])
    j = -1
    for t_m in market_list:
        f_market = open(directory_quote + '/' + t_m + ext_text, 'rU')
        flag = header
        i = 0
        for line in f_market.readlines():
            if flag:
                flag = not flag
                continue
            s = line.strip().split(delimiter)
            current_data[i][BASE_FEATURES + j] = float(s[4]) 
            i +=1
        f_market.close()
        j += 1
    flag = header
    i = 0
    for line in f_economic.readlines():
        if flag:
            flag = not flag
            continue
        s = line.strip().split(delimiter)
        current_data[i][NUM_MONTH-2] = float(s[4]) 
        i +=1    
    # get first entry, don't add to data
    previous_data = current_data[0]
    new_data = []
    for i in range(1, len(current_data)):
        c = current_data[i]
        if i < len(current_data) - 1:
           label = get_label(current_data[i][0], current_data[i+1][0])
        else:
           label = 0
        # don't add last entry
        if i != len(current_data) - 1: 
            l = []
            for i in range(NUM_MONTH):
                l.append(0.0)
            l[0] = label
            l[1] = c[1]
            l[2] = get_change(previous_data[2], c[2])
            l[3] = get_change(previous_data[3], c[3])
            l[4] = c[4]
            l[5] = get_change(previous_data[5], c[5])
            for i in range(BASE_FEATURES-1, NUM_MONTH-1):
                l[i] = get_change(previous_data[i], c[i])
            l[NUM_MONTH-1] = c[NUM_MONTH-1]
            new_data.append(l)
        previous_data = current_data[i]
    return new_data

def write_prep_data(ticker_name, data_list, directory_data = DEFAULT_DATA_DIR, ext_data = EXT_DATA):
    """
    just writes the list from prep_raw_data() to a data file
    """
    f_data = open(directory_data + '/' + ticker_name + ext_data, 'w')
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
    #return 1.0
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
def classify_sigmoid(v, weights, deviation = 0.001):
    probability = get_sigmoid(sum(v * weights))
    if probability > 0.5 + deviation: return  +1
    if probability < 0.5 - deviation: return -1
    return 0
    
#Stochastic Gradient Ascent (constant eta)
def SGA(data, label, eta = 0.001, num_iteration = 10):
    m,n = shape(data)
    weights = ones(n)   #initialize to all ones
    for j in range(num_iteration):
        for i in range(m):
            h = get_sigmoid(sum(data[i]*weights))
            error = label[i] - h
            weights = weights + eta * error * data[i]
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
        if c_s != int(c[0]):
            error_count += 1
    if data_count != 0:
        error_rate = (float(error_count)/data_count)
    print "error test rate on " + ticker_stock + " is: %f" % error_rate
    return error_rate

def logistic_multimonth_test(ticker_stock, same_month = False, directory_data = DEFAULT_DATA_DIR,\
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
        total_error += logistic_test(ticker_stock, directory_train_test, ext_train, ext_test)
    print 'average error rate (over %d tests--one/month) for %s is: %f' % ( num_test, ticker_stock, total_error/float(num_test))
    return total_error/float(num_test)

def logistic_multimonth_test_all(ticker_list = STOCK_LIST, same_month = False, directory_data = DEFAULT_DATA_DIR,\
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
       total_error += logistic_multimonth_test(t, same_month, directory_data, directory_train_test, ext_data, ext_train, ext_test)
    print 'average error rate across all tests (%d stocks) is: %f' % (num_test, total_error/float(num_test))  
