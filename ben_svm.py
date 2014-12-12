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

#num_prev = 5
def create_training_set(asset_name, num_prev, ff_nn=False):
    feat_dict, date_list = create_feats_dict(asset_name, 'nsdq')
    #num_prev = 5
    back_days = [None] * (num_prev + 1)
    #print date_list
    for date in date_list:
        for i in range(len(back_days) - 2, -1, -1):
            back_days[i + 1] = back_days[i]
        back_days[0] = feat_dict[date]
        if back_days[-1] == None: continue
        #print "BACKDAYS: DEBUG:", back_days[0]
        #print "DEBUG:", back_days[0]
        y = back_days[0][1]
        x_feats = []
        #x_feats.append(back_days[0][3])
        for i in range(1, len(back_days)):
            x_feats.extend(back_days[i][3:])
        num_feats = len(x_feats)

        if (ff_nn == True):
            x_feats = tuple(x_feats)
            if y == 1:
                y = (1, 0)
            else: y = (0, 1)
        #print "DEBUG:", x_feats
        if back_days[0][0] == 1: #or back_days[0][0] == 0: #This is a training item
            train_X.append(x_feats)
            train_Y.append(y)
        else:
            test_X.append(x_feats)
            test_Y.append(y)
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
def test_model(model, x, y, svm=False, verbose=False):
    #precision = corr / corr + false pos
    #recall = corr / corr + false neg

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
            if verbose == True: print "DEBUG------PREDICTED:", pred, "ACTUAL:", test_y, success
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
        #print "TRAIN SET:"
        #print "DEBUG-------LEN X:", len(train_X), "LEN_Y:", len(train_Y), "TEST_X:", len(test_X), "TEST_Y:", len(test_Y)
        #for i in range (0,5):
        #    print "DEBUG--------TRAIN X:", train_X[i], "TRAIN_Y", train_Y[i],
        test_model(ffnn, train_X, train_Y)
        #print "TEST SET:"
#        if _ == numEpochs - 1:
#            v = True
#        else:
#            v = False
        test_model(ffnn, test_X, test_Y)
        
    #trainer.trainEpochs(epochs=20)
    return ffnn

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

asset_name = "msft"
is_test_svm = False
is_rnn = False
num_prev_days_feats = 2
#only applicable if neural net
num_hidden_nodes = 10
num_sec_layer = -1
num_epochs = 20
learning_rate = 0.1
momentum = 0.0001
#only applicable for svm
kernel_type='linear'
#rbf, linear, poly

num_feats = create_training_set(asset_name, num_prev_days_feats, ff_nn=(not is_test_svm))
#print "NUM TRAIN:", len(train_X), "NUM TEST:", len(test_X)
if is_test_svm:
    model = build_svm(kernel_type)
else:
    model = ffnn(learning_rate, num_epochs, num_feats, 2, num_hidden_nodes, hidden2=num_sec_layer, rec=is_rnn, mom=momentum)

test_model(model, test_X, test_Y, svm=is_test_svm, verbose=True)
baseline_comp()
