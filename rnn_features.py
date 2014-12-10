import datetime

"""
 
Warning: There is no tests for file/directory existence
CSV files must be sorted in ascending order

"""
def dateInRange(date_val, date_range):
    """
    example usage: just FYI, used internally
    dateInRange('mm/dd/yyyy', ['mm/dd/yyyy', 'mm/dd/yyyy']) between 2 dates (inclusive)
    dateInRange('mm/dd/yyyy', ['mm/dd/yyyy', 0]) from date onward (0 arbitraty, any number will do)
    dateInRange('mm/dd/yyyy', ['mm']) filter for one month
    dateInRange('mm/dd/yyyy', ['yyyy']) filter for one year
    """
    d = datetime.datetime.strptime(date_val, "%m/%d/%Y")
    if len(date_range) == 2:
        d1 = datetime.datetime.strptime(date_range[0], "%m/%d/%Y")
        if isinstance(date_range[1], (int, long)):
             return (d >= d1 )
        d2 = datetime.datetime.strptime(date_range[1], "%m/%d/%Y")
        return (d >= d1 and d <= d2)
    else:
        if int(date_range[0]) <= 12:
            return (int(d.month) == int(date_range[0]))
        else:
            return (int(d.year) == int(date_range[0]))

def dateInMonth(date_val, month_val):
    # boolean test
    d = datetime.datetime.strptime(date_val, "%m/%d/%Y")
    return (int(d.month) == int(month_val[0]))

def dateInYear(date_val, year_val):
    # boolean test
    d = datetime.datetime.strptime(date_val, "%m/%d/%Y")
    return (int(d.year) == int(year_val[0]))
    
def getDayofWeek(string_date, delimiter = '/'):
    s = string_date.split(delimiter)
    d = datetime.date(int(s[2]), int(s[0]), int(s[1]))
    return datetime.date.weekday(d)
    
def getDate(string_date, delimiter = '/'):
    # convert date string into date object
    s = string_date.split(delimiter)
    d = datetime.date(int(s[2]), int(s[0]), int(s[1]))
    return d

def formatDate(string_date, delimiter = '/'):
    # convert date string from m/d/yy to mm/dd/yyyy
    return datetime.datetime.strptime(string_date,"%m/%d/%y").strftime('%m/%d/%Y')

def getDayofWeekVect(string_date, delimiter = '/'):
    # use value of day as index to list [0,0,0,0,0] and sets it to 1
    s = string_date.split(delimiter)
    d = datetime.date(int(s[2]), int(s[0]), int(s[1]))
    l = [0]*5
    l[int(datetime.date.weekday(d))] = 1
    return l
    
def getMonth(string_date, delimiter = '/'):
    s = string_date.split(delimiter)
    return int(s[0])
        
def getChangeRate(v1, v2):
    if v1 == 0: return 0
    return (v2 - v1) / float(v1)

def getAverage(v1, v2):
     return  (v1 + v2) / 2.0

STOCK_LIST = ['MSFT', 'GOOG', 'CSCO', 'IBM','AKAM', 'ADBE', 'AMZN', 'INTC', 'ORCL', 'NVDA']
MARKET_LIST = ['^COMP', '^DJIA', '^GSPC']
ECONOMIC_LIST = ['^W5000']
BASE_FEATURES = 6

""""
Usage example: 

d = DataFactory()      # all values have defaults
d.labelType = 'E'
d.uSensitivity = 0.01 (default 0.001)
d.quoteDir = './quotes'
d.dataDir = './data'
d.createDataset('IBM') 

labelList, featureList = d.loadDataSet('IBM', ['2014']) # loads all values for 2014

#####################################################################################################-
N.B.: after calling d.loadDataSet() an internal list is saved in the class
use d.saveTrainData() or d.saveTestData() to save your loaded selection as Train or Test 
#####################################################################################################-
interesting properties:
   labelType = (values: 'S', 'A', 'E')
      'S' : y-label is 0, 1
      'A' : y-label is 1, -1
      'E' : y-label is 1, 0, -1
   uSensitivity = (default: 0.001) works only with labelType == 'E' to determine interval for unchanged values
      
   stockList = list of stocks (default STOCK_LIST)
   marketList = list of stocks (default STOCK_LIST)  # you can limit or increase the number of market indices
   economicList = list of stocks (default STOCK_LIST)
   headerCount skips line reading in csv file by its value
other properties:
   xxxDir sets directory
   xxxExt sets file extension
   _xxxx  internal usage
#####################################################################################################-
IMPORTANT:
  if you change labelType, uSensitiviy, marketList or economicList you have to re-run createDataSet()
  to regenerate the data
"""
class DataFactory:
    def __init__(self, label_type='S', header_count=1, u_sensitivity=0.001, quote_dir='./quotes',\
                 data_dir ='./data', train_dir='./train', test_dir='./train', delimiter=',', quote_ext ='csv',\
                 train_ext='train', data_ext='data', test_ext='test'):
        self.labelType = label_type.upper()
        if self.labelType != 'S' or self.labelType != 'A' or self.labelType != 'E':
            self.labelType = 'S'
        self.quoteDir = quote_dir
        self.dataDir = data_dir
        self.trainDir = train_dir
        self.testDir = test_dir
        self.delimiter = delimiter
        self.quoteExt = quote_ext
        self.trainExt = train_ext
        self.dataExt = data_ext
        self.testExt = test_ext
        self.headerCount = header_count
        self.uSensitivity = u_sensitivity
        self.stockList = STOCK_LIST
        self.marketList = MARKET_LIST
        self.economicList = ECONOMIC_LIST
        self._tickerLoad = ''
        self._dataLoad = []
    
    def getFileName(self, file_type, file_root):
        param = file_type.upper()
        if param == 'Q':
            return self.quoteDir + '/' +  file_root + '.' + self.quoteExt
        if param == 'D':
            return self.dataDir + '/' +  file_root + '.' + self.dataExt
        if param == 'T':
            return self.testDir + '/' +  file_root + '.' + self.testExt
        if param == 'TR':
            return self.trainDir + '/' +  file_root + '.' + self.trainExt
            
    def getLabel(self, v1, v2):
        c = getChangeRate(v1, v2)
        if self.labelType == 'S':
            if c >= 0:
                return +1
            else:
                return 0
        elif self.labelType == 'A':
            if c >= 0:
                return +1
            else:
                return -1
        elif self.labelType == 'E':
            if c > self.uSensitivity:
                return +1
            if c < -self.uSensitivity:
                return -1
            return 0
            
    def createDataSet(self, ticker_symbol):
        fStock = open(self.getFileName('Q', ticker_symbol), 'rU')
        featureCount = BASE_FEATURES + len(MARKET_LIST) + len(ECONOMIC_LIST)
        featureCount += 6     # 1 for label, 5 for Day of Week Vector
        
        #Files must have same number of entries in ascending date order
        
        currentData= []
        k = 0
        for line in fStock.readlines():
            if k < self.headerCount:
                k += 1; continue
            s = line.strip().split(self.delimiter)
            l = [0] * featureCount
            l[0] = formatDate(s[0])           # 0 format date mm/dd/yyyy
            l[1] = 0                          # 1 label (to be set later)
            v = getDayofWeekVect(l[0])
            l[2] = v[0]                       # 2-6 dense vector for Day of Week 
            l[3] = v[1]
            l[4] = v[2]
            l[5] = v[3]
            l[6] = v[4]
            l[7] = float(s[1])                # 7 Open
            l[8] = float(s[4])                # 8 Close
            l[9] = float(s[6])                # 9 Adj. Close
            l[10] = float(s[2]) - float(s[3]) # 10 High - Low
            l[11] = float(s[5])               # 11 volume
            currentData.append(l)
        j = 12
        extendedMarket = self.marketList + self.economicList
        for m in extendedMarket:
            fMarket = open(self.getFileName('Q', m), 'rU')
            k = 0
            i = 0
            for line in fMarket.readlines():
                if k < self.headerCount:
                    k += 1; continue
                s = line.strip().split(self.delimiter)
                currentData[i][j] = float(s[4]) 
                i +=1
            fMarket.close()
            j += 1
        
        # get first entry, don't add to data
        previousData = currentData[0]
        newData = []
        for i in range(1, len(currentData)):
            if i < len(currentData) - 1:
                label = self.getLabel(currentData[i][8], currentData[i+1][8])
            else:
                label = 0
            # don't add last entry
            if i != len(currentData) - 1: 
                l = [0] * featureCount
                l[0] = currentData[i][0]
                l[1] =  label
                l[2] = currentData[i][2]
                l[3] = currentData[i][3]
                l[4] = currentData[i][4]
                l[5] = currentData[i][5]
                l[6] = currentData[i][6]
                for j in range(BASE_FEATURES, featureCount):
                    l[j] = getChangeRate(previousData[j], currentData[i][j])
                newData.append(l)
            previousData = currentData[i]
        self.saveDataSet(ticker_symbol, newData)
        
    def _save(self, dataset_type, ticker_symbol, data_list):
        fData = open(self.getFileName(dataset_type, ticker_symbol), 'w')
        for i in range(len(data_list)):
            line = ' '.join(str(x) for x in data_list[i]) + '\n'
            fData.write(line)
    
    def saveDataSet(self, ticker_symbol, data_list):
        self._save('D', ticker_symbol, data_list)
        
    def saveTrainSet(self):
        self._save('TR', self._tickerLoad, self._dataLoad)
        
    def saveTestSet(self):
        self._save('T', self._tickerLoad, self._dataLoad)
    
    def loadDataSet(self, ticker_symbol, date_range=None):
        self._tickerLoad = ticker_symbol
        fData = open(self.getFileName('D', ticker_symbol), 'rU')
        featureList = []
        labelList = []
        self._dataLoad = []
        if date_range is not None:
            func = dateInRange
        else:
            func = lambda x, y : True
        for line in fData.readlines():
            s = line.strip().split(' ')
            if func(s[0], date_range):
                for i in range(1, len(s)):
                    s[i] = float(s[i])
                labelList.append(s[1])
                featureList.append(s[2:])
                self._dataLoad.append(s[1:])
       
        return labelList, featureList
