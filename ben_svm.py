from sklearn import svm, cross_validation
from feat_factory import *
import math
from pybrain.datasets import *
#from pybrain.datasets import ClassificationDataset
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer, TanhLayer, LinearLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import RecurrentNetwork
import matplotlib.pyplot as plt

#squared error
def error(pred, act):
    return (pred - act) ** 2


#asset_dict:   date --> feat_list
#feat_list: [train, label1, label2, features ---]
#generate x features list for 5 days back

train_X = []
train_Y = []
test_X = []
test_Y = []
#pred_y is 0 for long, 1 for nothing, 2 for short.
#date --> [x, y, act_adj_delta, pred_y, succ-code]
date_vis_dict = {}

#num_prev = 5
def create_training_set(asset_name, num_prev, ff_nn=False, three_class=False):
    feat_dict, date_list = create_feats_dict(asset_name, 'nsdq')
    #num_prev = 5
    back_days = [None] * (num_prev + 1)
    #print date_list
    for date in date_list:
        for i in range(len(back_days) - 2, -1, -1):
            back_days[i + 1] = back_days[i]
        back_days[0] = feat_dict[date]
        if back_days[-1] == None: continue

        if three_class == False:
            y = back_days[0][1]
        else:
            y = tuple([back_days[0][1], back_days[0][2]])
            
        date_vis_dict[date] = []
        x_feats = []
        for i in range(1, len(back_days)):
            x_feats.extend(back_days[i][3:])
        num_feats = len(x_feats)

        if (ff_nn == True):
            x_feats = tuple(x_feats)
            if three_class == False:
                if y == 1:
                    y = (1, 0)
                else: y = (0, 1)
            else:
                first = y[0]
                third = y[1]
                if first + third < 1:
                    sec = 1
                else:
                    sec = 0
                y = (first, sec, third)
                #print "DEBUG:::::", y
        #print "DEBUG:", x_feats
        actual_adj_delta = back_days[0][3]
        #print "DEBUG::::::", back_days[0], "ACT ADJ DELTA", actual_adj_delta
        #print "DEBUG::: X:", x_feats, "Y:", y
        date_vis_dict[date].append(x_feats)
        date_vis_dict[date].append(y)
        date_vis_dict[date].append(actual_adj_delta)
        if back_days[0][0] == 1: #or back_days[0][0] == 0: #This is a training item
            train_X.append(x_feats)
            train_Y.append(y)
#            date_vis_dict[date].append(1)
        else:
            test_X.append(x_feats)
            test_Y.append(y)
#            date_vis_dict[date].append(0)
    return num_feats

def build_svm(kern):
    #print "DEBUG Y-----:", train_Y
    #model = svm.SVC()
    #cross_validation.cross_val_score(model, X, y, scoring='mean_squared_error')
    clf = svm.SVC(kernel=kern)
    clf.fit(train_X, train_Y) 
    return clf
#    tr_err = clf.score(train_X, train_Y)
#    test_err = clf.score(test_X, test_Y)

    #for x, y in zip(test_X, test_Y):
        #print "PRED:", clf.predict([x]), "ACTUAL:", y

    #print "TRAINING ERROR:", tr_err_
    #print "TESTING ERROR:", test_err

#def ff_error_check(ffnn):
#    for test_x, test_y in zip(test_X, test_Y):
#        pred = ffnn.activate(test_X)
        #print "FFNN PRED:----", pred, "ACT:", test_Y[0]

def test_multiclass(model, x, y, svm=False, verbose=False):
    corr = 0.
    false_pos = 0.
    false_neg = 0.
    epic_fail = 0.

#Codes: 0 = success; 1 = fail; 2 = epic fail; 3 = false_neg; 4 = true_negative
#    for test_x, test_y in zip(x, y):
    for date , val in date_vis_dict.iteritems():
        #print "DATE:", date, "VAL", val
        test_x = val[0]
        test_y = val[1]
#        test_x, test_y, is_train = val
        success = ""
        if svm == False:
            pred = model.activate(test_x)
            max_index = -1
            max_val = -99
            for i in range(0, len(pred)):
                if pred[i] > max_val:
                    max_index = i
                    max_val = pred[i]
            pred_y = max_index
            date_vis_dict[date].append(pred_y)
            act_y = -1
            for i in range(0, len(test_y)):
                if test_y[i] == 1:
                    act_y = i
            if pred_y == 2 or pred_y == 0: 
                if pred_y - act_y == 0:
                    corr += 1
                    success = "SUCCESS"
                    date_vis_dict[date].append(0)
                elif math.fabs(pred_y - act_y) == 1:
                    false_pos += 1
                    success = "\t\tfalse_pos_fail"
                    date_vis_dict[date].append(1)
                elif math.fabs(pred_y - act_y) == 2:
                    epic_fail += 1
                    success = "\t\t\t\tFP EPIC FAIL"
                    date_vis_dict[date].append(2)
            elif pred_y == 1 and (act_y == 0 or act_y == 2):
                false_neg += 1
                date_vis_dict[date].append(4)
            else:
                date_vis_dict[date].append(5)
    tot = corr + false_pos + epic_fail
    corr_p = corr / tot
    fail_p = false_pos / tot
    epic_fail_p = epic_fail / tot
    recall = corr / (corr + false_neg)
    print "CORR", corr, "FAIL", false_pos, "EPIC_FAIL", epic_fail, "CORR_P", corr_p, "FAIL_P", fail_p, "E_FAIL_P", epic_fail_p, "RECALL", recall
    return corr_p

def test_model(model, x, y, svm=False, verbose=False, three_class=False):
    corr = 0.
    false_pos = 0.
    false_neg = 0.
    true_neg = 0.

    for test_x, test_y in zip(x, y):
        #print "DEBUG TEST MODEL:", pred, "TEST Y", test_y
        success = ""
        if svm == False: #THIS IS NEURAL NETWORK
            pred = model.activate(test_x)
            if pred[0] > pred[1] and test_y[0] == 1:
                success = "SUCCESS"
                corr+= 1
            elif pred[0] > pred[1] and test_y[0] == 0:
                false_pos+= 1
                success = "\t\tfalse pos"
            elif pred[1] >= pred[0] and test_y[0] == 1:
                false_neg+= 1
                success = "\t\tfalse neg"
            else:
                true_neg += 1 
                success = "SUCC NEG"
            #if verbose == True: print "DEBUG------PREDICTED:", pred, "PRED", pred_y, "ACTUAL:", act_y, success
        else: #THIS IS SVM
            pred = model.predict(test_x)
            if pred == 1 and test_y == 1:
                corr+= 1
                success = "SUCCESS"
            elif pred == 1 and test_y == 0:
                false_pos+= 1
                success = "\t\tfalse pos"
            elif pred == 0 and test_y == 1:
                false_neg+= 1
                success = "\t\tfalse neg"
            else:
                true_neg += 1
                success = "SUCC NEG"
            if verbose == True: print "DEBUG------PREDICTED:", pred, "ACTUAL:", test_y, success
        #print "PRED:", pred, "ACT:", test_y
    #print "CORR:", corr, "FP:", false_pos, "FN:", false_neg
    prec = corr / (corr + false_pos)
    recall = corr / (corr + false_neg)
    f_score = 2. * (prec * recall) / (prec + recall)
    err = (corr + true_neg) / (corr + true_neg + false_pos + false_neg)
    print "CORR", corr, "FP", false_pos, "FN", false_neg, "PRECISION:", prec, "RECALL", recall, "F1 SCORE:", f_score, "NON-ERR RATE:", err
    return f_score
            
def build_2ffnn(inp, h1, h2, out):
    n = FeedForwardNetwork()
    inLayer = LinearLayer(inp)
    hiddenLayer1 = TanhLayer(h1)
    hiddenLayer2 = TanhLayer(h2)
    outLayer = LinearLayer(out)
    #outLayer = SoftmaxLayer(out)
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer1)
    n.addModule(hiddenLayer2)
    n.addOutputModule(outLayer)
    in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
    hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
    hidden2_to_out = FullConnection(hiddenLayer2, outLayer)
    n.addConnection(in_to_hidden1)
    n.addConnection(hidden1_to_hidden2)
    n.addConnection(hidden2_to_out)
    n.sortModules()
    n.randomize()
    return n

def build_rec(inp, hid, out):
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(inp, name='in'))
    n.addModule(TanhLayer(hid, name='hidden'))
    n.addOutputModule(SoftmaxLayer(out, name='out'))
    n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
    n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
    n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))
    n.sortModules()
    #n.randomize()

    return n

def ffnn(lr, num_epochs, inp, out, hidden, hidden2=-1, rec=False, mom=0.0):
    ds = ClassificationDataSet(inp, out)
    for tr_x, tr_y in zip(train_X, train_Y):
        #print "DEBUG:", tr_x, tr_y
        ds.addSample(tr_x, tr_y)
    if hidden2 != -1:
        ffnn = build_2ffnn(inp, hidden, hidden2, out)
    elif rec==True:
        ffnn = build_rec(inp, hidden, out)
    else:
        ffnn = buildNetwork(inp, hidden, out, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(ffnn, ds,learningrate = lr, momentum=mom, weightdecay=0.0, verbose=True)
    
    
    ######
    numEpochs = num_epochs
    for _ in range(0, numEpochs):
        trainer.train()

        if not triple_class:
            test_model(ffnn, train_X, train_Y)
            test_model(ffnn, test_X, test_Y)
#        else:
#            test_multiclass(ffnn, train_X, train_Y)
#            test_multiclass(ffnn, test_X, test_Y)
    return ffnn

def visualize():
    #create histogram
    min_month_val = 9999999.
    max_month_val = 0
    max_val = 0
    data_hist_dict = {}
    for date, val in date_vis_dict.iteritems():
        #print "DEBUG DATE", date
        yr = date.year
        mo = date.month - 1
        #print "DEBUG YR:", yr, "MO:", mo
        date_val = float(yr) + float(mo) / 12
        if date_val < min_month_val:
            min_month_val = date_val
        if date_val > max_month_val:
            max_month_val = date_val
        if date_val not in data_hist_dict:
            data_hist_dict[date_val] = {}
            for i in range(0, 6):
                data_hist_dict[date_val][i] = 0
        data_hist_dict[date_val][val[4]] += 1
        if data_hist_dict[date_val][val[4]] > max_val:
            max_val = data_hist_dict[date_val][val[4]] 
    
    #plot
    succ = [[],[]]
    fail = [[],[]]
    e_fail = [[],[]]
    by_year_dict = {}
    by_year_perc = {}
    for month_val, val in data_hist_dict.iteritems():
        #print "VAL:", val
        for code, value in val.iteritems():
#            print "DEBUG: CODE", code, "VALUE:", value
            if int(month_val) not in by_year_dict:
                by_year_dict[int(month_val)] = [0, 0, 0]
            if code > 2: continue
            #print "DEBUG",  by_year_dict[int(month_val)][code]
            by_year_dict[int(month_val)][code] += value
#            print "DEBUG CODE:", code, "VALUE:", value
            if code == 0:
                succ[0].append(month_val)
                succ[1].append(value)
            if code == 1:
                fail[0].append(month_val)
                fail[1].append(value)
            if code == 2:
                e_fail[0].append(month_val)
                e_fail[1].append(value)
    for k, v in by_year_dict.iteritems():
        if k not in by_year_perc:
            by_year_perc[k] = [0.,0.,0.]
        tot = sum(x for x in by_year_dict[k])
        for index, count in enumerate(v):
            by_year_perc[k][index] = float(count) / tot

    succ_p = [[],[]]
    fail_p = [[],[]]
    e_fail_p = [[],[]]

    for k, v in by_year_perc.iteritems():
        for i in range(0, len(v)):
            if i == 0:
                succ_p[0].append(k)
                succ_p[1].append(v[i])
            if i == 1:
                fail_p[0].append(k)
                fail_p[1].append(v[i])
            if i == 2:
                e_fail_p[0].append(k)
                e_fail_p[1].append(v[i])

    plt.plot(succ_p[0], succ_p[1], 'g--')
    plt.plot(fail_p[0], fail_p[1], 'y--')
    plt.plot(e_fail_p[0], e_fail_p[1], 'r--')
    plt.axis([int(min_month_val), int(max_month_val) + 1, 0, 1])
    plt.title("MSFT Classification % Success Over Time")
    plt.ylabel('%')
    plt.xlabel('Year')
    plt.show()

    
    """
    plt.plot(succ[0], succ[1], 'go')
    plt.plot(fail[0], fail[1], 'yo')
    plt.plot(e_fail[0], e_fail[1], 'ro')
    plt.axis([int(min_month_val), int(max_month_val) + 1, 0, max_val + 5,])
    plt.title("MSFT Classification Success Over Time")
    plt.ylabel('# Hits')
    plt.xlabel('Year')
    plt.show()
    """
    return data_hist_dict, min_month_val

def profit():
    roi_by_month = {}
    cum_roi_by_month = {}
    min_month = 2015
    max_month = 1990
    for date, val in date_vis_dict.iteritems():
        date_val = date.year + (date.month - 1) / 12.
        #print "DEBUG DATE VAL:", date_val
        if date_val < min_month:
            min_month = date_val
        if date_val > max_month:
            max_month = date_val

        if date_val not in roi_by_month:
            roi_by_month[date_val] = 0
            #cum_roi_by_month[date_val] = 0
        pred_y = val[3]
        delta = val[2] #double check this
#        print "DEBUG::: pred_y", pred_y, "DELTA:", delta
        if pred_y == 0: #long
            roi_by_month[date_val] += delta
        elif pred_y == 2: #short
            roi_by_month[date_val] -= delta
    
    for k, v in roi_by_month.iteritems():
        print "TIME:", k, "ROI:", v    
    cum_roi = 0
    min_cum = 0
    max_cum = 0
    t = min_month
    for k in sorted(roi_by_month.keys()):
        cum_roi += roi_by_month[k]
        cum_roi_by_month[k] = cum_roi

        if cum_roi > max_cum:
            max_cum = cum_roi
        if cum_roi < min_cum:
            min_cum = cum_roi
#    while t <= max_month:
#        if t in roi_by_month:
#            
#            cum_roi += roi_by_month[t]
#            print "DEBUG TIME:", t, "CUM_ROI:", cum_roi
#            if cum_roi > max_cum:
#                max_cum = cum_roi
#            if cum_roi < min_cum:
#                min_cum = cum_roi
#        cum_roi_by_month[t] = cum_roi
#        t += 1. / 12
    x = []
    y = []
    print "MIN MONTH:", min_month
    print "BY MONTH:", roi_by_month
    for k, v in cum_roi_by_month.iteritems():
        #print "TIME:", k, "CROI:", v
        x.append(k)
        y.append(v)
    
    plt.plot(x, y, 'go')
    plt.axis([int(min_month), int(max_month) + 1, min_cum, max_cum])
    plt.title("ROI Basic Trading Strategy, FFNN for MSFT")
    plt.ylabel("Cumulative ROI %")
    plt.xlabel("Year")
    plt.show()
        
#compare to sample success stat
def baseline_comp():
    succ = 0.

    for y in test_Y:
        if not is_test_svm:
            if y[0] == 1:
                succ += 1
        else:
            if y == 1:
                succ += 1
    print "BASELINE HIT RATE:", succ / len(test_Y)
########Parameter

asset_name = "aapl"
is_test_svm = False
is_rnn = False
num_prev_days_feats = 3
#only applicable if neural net
num_hidden_nodes = 20
num_sec_layer = -1

num_epochs = 10
learning_rate = 0.01
momentum = 0.0005

#only applicable for svm
kernel_type='linear'
#rbf, linear, poly
triple_class = True
if triple_class == True:
    num_out = 3
else:
    num_out = 2

num_feats = create_training_set(asset_name, num_prev_days_feats, ff_nn=(not is_test_svm), three_class=triple_class)
#print "NUM TRAIN:", len(train_X), "NUM TEST:", len(test_X)
if is_test_svm:
    model = build_svm(kernel_type)
else:
    model = ffnn(learning_rate, num_epochs, num_feats, num_out, num_hidden_nodes, hidden2=num_sec_layer, rec=is_rnn, mom=momentum)

test_multiclass(model, test_X, test_Y, svm=is_test_svm, verbose=True)
baseline_comp()
#visualize()
profit()
