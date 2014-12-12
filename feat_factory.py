import csv
import datetime
import re

def get_weekday(str_date):

    #raw_date = row[0]
    regex = re.compile("(\d+)-(\d+)-(\d+)")
    r = regex.search(str_date)
    year, month, date = r.groups()
    date = datetime.date(int(year), int(month), int(date))
    day = date.weekday()     
    return day, date

#def gen_label(open, adj_close):
#    if adj_close > open: return 1
#    else: return 0

def is_train(date):
  
    if (date.year + date.day) % 2 == 1: return 1
    else: return 0

#def gen_features(Date, Open, High, Low, Close, Volume, Adj_Close, weekday):
path_dict = {'msft': 'quotes/MSFT.csv',
             'nsdq': 'quotes/^IXIC.csv',
             'msft_rec': 'quotes/MSFT_rec.csv'}


def create_feats_dict(str_asset, index):
    date_list = []
    feat_dict = {}
    all_feats_dict = {}
    index_dict = {}
    ret_date_list = []
    path = path_dict[str_asset]
    path_index = path_dict[index]
    with open(path, 'rU') as csvfile, open(path_index, 'rU') as indexFile:
        r = csv.reader(csvfile, delimiter=',')
        ri = csv.reader(indexFile, delimiter=',')
        for row in ri:
            if row[0] == "Date": continue
            str_date = row[0]
            weekDay, date = get_weekday(str_date)
            Date, Open, High, Low, Close, Volume, Adj_Close = row
            adj_close = float(Adj_Close)
            index_dict[date] = adj_close
#        num_raw_fin_feats = 7
#######
#        #pos_delta, neg_delta, open, hi, lo, vol, adj_close
#        max = [float("-inf")] * num_raw_fin_feats
#        min = [float("inf")] * num_raw_fin_feats
#        print max, min
#######
        max_open, min_open = float("-inf"), float("inf")
        max_hi, min_hi = float("-inf"), float("inf")
        max_lo, min_lo = float("-inf"), float("inf")
        max_vol, min_vol = float("-inf"), float("inf")
        max_adj_close, min_adj_close = float("-inf"), float("inf")    
        #max_pos_delta, min_pos_delta = float("-inf"), float("inf")
        #max_neg_delta, min_neg_delta = float("-inf"), float("inf")
        
        for row in r:
            if row[0] == "Date": continue
            str_date = row[0]
            weekday, date = get_weekday(str_date)
        #date_list.append(date)
           
            Date, Open, High, Low, Close, Volume, Adj_Close = row
            close = float(Close)
            _open = float(Open)
            if _open > max_open: max_open = _open
            if _open < min_open: min_open = _open
            hi = float(High)
            if hi > max_hi: max_hi = hi
            if hi < min_hi: min_hi = hi
            lo = float(Low)
            if lo > max_lo: max_lo = lo
            if lo < min_lo: min_lo = lo
            vol = float(Volume)
            if vol > max_vol: max_vol = vol
            if vol < min_vol: min_vol = vol
            adj_close = float(Adj_Close)
            if adj_close > max_adj_close: max_adj_close = adj_close
            if adj_close < min_adj_close: min_adj_close = adj_close
            #This is for classification
            if hi - _open > 1:
                pos_delta = 1
            else: pos_delta = 0
            if _open - lo > 1:
                neg_delta = 1
            else: neg_delta = 0
            #pos_delta = hi - _open
            
#            if pos_delta > max_pos_delta: max_pos_delta = pos_delta
#            if pos_delta < min_pos_delta: min_pos_delta = pos_delta
            #neg_delta = _open - lo
#            if neg_delta > max_neg_delta: max_neg_delta = neg_delta
#            if neg_delta < min_neg_delta: min_neg_delta = neg_delta


            date_list.append(date)
            feat_dict[date] = [is_train(date), _open, hi, lo, vol, close, adj_close, pos_delta, neg_delta]


        

    #for normalization
            range_open = max_open - min_open
            range_hi = max_hi - min_hi
            range_lo = max_lo - min_lo
            range_vol = max_vol - min_vol
            range_adj_close = max_adj_close - min_adj_close
#            range_pos_delta = max_pos_delta - min_pos_delta
#            range_neg_delta = max_neg_delta - min_neg_delta
            

        prevDate = None
        currDate = None
        for date in date_list:
            currDate = prevDate
            prevDate = date
            if prevDate == None or prevDate not in index_dict or currDate not in index_dict:
                continue
            else:
                ret_date_list.append(currDate)
                #print "PREV:", prevDate, "CURR:", currDate
                adj_delta = 100* (feat_dict[currDate][5] - feat_dict[prevDate][5]) / feat_dict[prevDate][5]
                ind_delta = 100* (index_dict[currDate] - index_dict[prevDate]) / index_dict[prevDate]
            all_feats = []
            tr, o, h, l, v, c, ac, pd, nd = feat_dict[date]
            #(pd - min_pos_delta) / range_pos_delta, (nd - min_neg_delta) / range_neg_delta,
            #fin_feat_list = [pd, nd, adj_delta, ind_delta]
            if adj_delta > 1:
                y_out = 1
            else:
                y_out = 0
            hi_delta = h - o
            lo_delta = l - o
            fin_feat_list = [tr, y_out, nd, adj_delta, ind_delta, hi_delta, lo_delta]
#            fin_feat_list = [tr, y_out, nd,
#                             (o - min_open) / range_open, (h - min_hi) / range_hi, (l - min_lo) / range_lo, 
#                             (v - min_vol) / range_vol, adj_delta, ind_delta]
                       
            date_feat_list = []
        #Add Feature Template for Day of Week and Month
            weekday = date.weekday()
            month = date.month
            for i in [0,4]:
                if weekday == i: date_feat_list.append(1)
                else: date_feat_list.append(0)
            
#            for i in [0,11]:
#                if month == i: date_feat_list.append(1)
#                else: date_feat_list.append(0)
            all_feats = []
#            all_feats = [is_train(date)]
            all_feats.extend(fin_feat_list)

#            all_feats.extend(date_feat_list)
            all_feats_dict[currDate] = all_feats

    #print date_list
#    print all_feats_dict
    #print ret_date_list
    return all_feats_dict, ret_date_list

#dict, date_list = create_feats_dict('msft', 'nsdq')
#print dict
