import sys, os

# SAMPLE USAGE:
# python3 meta.py leakyrelu-0.3 1 Adagrad 2 0.548817752164406 472 0.01019964742517334 he_uniform SUBJ 0.01

sys.path.append("..")

from neuralnets.MLPs import MLPbasic
from handleHyper import getInitializer 
from loadData import getPaths,makeSplits,readSubj,loadTrainDevTest

dataset = '/work/scratch/se55gyhe/Act_func/outputs/' #TREC'


##################
##################
################## First we handle the hyperparams 
##################
##################
functions = sys.argv[1]
index=sys.argv[2]
opt = sys.argv[3] # random.choice(['Adam',"RMSprop","Adagrad","Adadelta","Adamax","Nadam","sgd"])
layers = int(sys.argv[4]) #random.choice([1, 2, 3, 4])
dropout_values = float(sys.argv[5]) #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.75])
units = int(sys.argv[6]) # random.choice([50, 100, 200, 300, 400, 500])
learning_rate = float(sys.argv[7]) # random.choice([0.002,0.001,0.005,0.01,0.1,0.5,1.0])
init_name = sys.argv[8] # he_normal, rnormal, etc. 
mydata = sys.argv[9] # TREC,SUBJ,MR,PE-sent2vec,PE-infersent
dataset = dataset+mydata
try:
  trainRatio = float(sys.argv[10])
except IndexError:
  trainRatio = 0.5
devRatio = trainRatio + 0.1

init,optimizer,functions,maxout_k = getInitializer(init_name,learning_rate,opt,functions)

####
#### load data
####
cbm = "loss" 
datasize="full"

if mydata=="TREC":
  base,trainX,trainY,testX,testY,trainRatio = getPaths(mydata)
  train_X,train_Y,dev_X,dev_Y,test_X,test_Y = makeSplits(base,trainX,trainY,trainRatio,testX,testY)

elif mydata=="SUBJ":
  data0 = '/work/scratch/se55gyhe/Act_func/sentence_classification/02_Subj/dataset/quote.tok.gt9.5000-vecs'
  data1 = '/work/scratch/se55gyhe/Act_func/sentence_classification/02_Subj/dataset/plot.tok.gt9.5000-vecs'
  train_X,train_Y,dev_X,dev_Y,test_X,test_Y = readSubj(data0,data1,trainRatio,devRatio)  
  datasize=str(trainRatio)

elif mydata=="MR":
  data0 = '/work/scratch/se55gyhe/Act_func/sentence_classification/01_MR/dataset/rt-polarity.pos.vecs'
  data1 = '/work/scratch/se55gyhe/Act_func/sentence_classification/01_MR/dataset/rt-polarity.neg.vecs'
  train_X,train_Y,dev_X,dev_Y,test_X,test_Y = readSubj(data0,data1, trainRatio,devRatio)
  datasize=str(trainRatio)

elif mydata.startswith("PE"):
  base,trainX,trainY,devX,devY,testX,testY = getPaths(mydata)
  train_X,train_Y,dev_X,dev_Y,test_X,test_Y = loadTrainDevTest(base,trainX,trainY,devX,devY,testX,testY)
  cbm = "f1"

#################

print(functions,opt,layers,dropout_values,units,learning_rate,init_name)
sys.stdout.flush()

MLPbasic(train_X, train_Y, dev_X, dev_Y, test_X, test_Y, dataset, functions, (opt,optimizer), layers,
         dropout_values, units, (init_name,init),index,learning_rate,maxout_k=maxout_k,callbacks_metric=cbm,datasize=datasize)
