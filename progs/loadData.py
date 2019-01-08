import sys,numpy as np

sys.path.append("/work/scratch/se55gyhe/Act_func/")

from utils.data import read_data, load_labels_trec, merge, load_labels_pe

def makeSplits(base,trainX,trainY,trainRatio,testX,testY):

 train_dev_data = read_data(base+trainX)
 train_dev_labels = load_labels_trec(base+trainY)

 print("shape of train and dev data ", train_dev_data.shape)
 print("shape of train and dev labels ", train_dev_labels.shape)
 print()

 data_length = train_dev_data.shape[0]

 train_X = train_dev_data[:int(trainRatio * data_length)]
 train_Y = train_dev_labels[:int(trainRatio * data_length)]

 dev_X = train_dev_data[int(trainRatio * data_length):]
 dev_Y = train_dev_labels[int(trainRatio * data_length):]

 test_X = read_data(base+testX)
 test_Y = load_labels_trec(base+testY)

 print("shape of train data ", train_X.shape)
 print("shape of train labels ", train_Y.shape)
 print()

 print("shape of test data ", test_X.shape)
 print("shape of test labels ", test_Y.shape)
 print()

 return train_X,train_Y,dev_X,dev_Y,test_X,test_Y

def loadTrainDevTest(base,trainX,trainY,devX,devY,testX,testY):
 train_X = read_data(base+trainX)
 train_Y = load_labels_pe(base+trainY)

 dev_X = read_data(base+devX)
 dev_Y = load_labels_pe(base+devY)

 test_X = read_data(base+testX)
 test_Y = load_labels_pe(base+testY)

 # train
 print("train data shape ", train_X.shape)
 print("train labels shape ", train_Y.shape)
 print()
 # dev
 print("dev data shape ", dev_X.shape)
 print("dev labels shape ", dev_Y.shape)
 print()
 # test
 print("test data shape ", test_X.shape)
 print("test labels shape ", test_Y.shape)

 return train_X,train_Y,dev_X,dev_Y,test_X,test_Y


def getPaths(mydata):

 if mydata=="TREC":
  base='/work/scratch/se55gyhe/Act_func/sentence_classification/03_Trec/'
  trainX="dataset/train_5500-sents-vecs"
  trainY="dataset/train_5500-labels"
  testX="dataset/TREC_10-sents-vecs"
  testY="dataset/TREC_10-labels"
  trainRatio=0.85
 elif mydata=="SUBJ":
  base="/work/scratch/se55gyhe/Act_func/sentence_classification/02_Subj/" 
  trainX="dataset/quote.tok.gt9.5000-vecs"  
  trainY="dataset/plot.tok.gt9.5000-vecs"
  testX=None
  testY=None
 elif mydata.startswith("PE"):
  _,emb = mydata.split("-")
  base="/work/scratch/se55gyhe/Act_func/sentence_classification/04_PE/" # originally
  base="/home/se55gyhe/ArgMin/"
  trainX='%s/dataset/student-essays-train-vecs'%emb
  trainY='%s/dataset/student-essays-train-labels.txt'%emb
  devX='%s/dataset/student-essays-dev-vecs'%emb
  devY='%s/dataset/student-essays-dev-labels.txt'%emb
  testX='%s/dataset/student-essays-test-vecs'%emb
  testY='%s/dataset/student-essays-test-labels.txt'%emb
  trainRatio=None

 if trainRatio is None:
   return base,trainX,trainY,devX,devY,testX,testY
 
 return base,trainX,trainY,testX,testY,trainRatio

def readSubj(data0,data1,trainRatio,devRatio):
 subj_data = read_data(data0)
 subj_labels = np.repeat([[1, 0]], subj_data.shape[0], axis=0)
 # Objective data
 obj_data = read_data(data1)
 obj_labels = np.repeat([[0, 1]], obj_data.shape[0], axis=0)

 print("DATA READ"); sys.stdout.flush()


 # Shapes
 print("shape of positive data ", subj_data.shape)
 print("shape of positive labels ", subj_labels.shape)
 print()
 print("shape of negative data ", obj_data.shape)
 print("shape of negative labels ", obj_labels.shape)
 print()

 # unite data
 data = merge(subj_data, obj_data)
 labels = merge(subj_labels, obj_labels)

 # randomly shuffle data and labels
 np.random.seed(7) # always the same split
 shuffle_indices = np.random.permutation(np.arange(len(data)))
 data_shuffled = data[shuffle_indices]
 labels_shuffled = labels[shuffle_indices]

 data_len = data.shape[0]
 train_index = int(trainRatio * data_len)
 dev_index = int(devRatio * data_len)
 test_index = data_len

 train_X = data[:train_index]
 dev_X = data[train_index:dev_index]
 test_X = data[train_index:test_index]


 train_Y = labels[:train_index]
 dev_Y = labels[train_index:dev_index]
 test_Y = labels[train_index:test_index]

 print("shape of train data ", train_X.shape)
 print("shape of train labels ", train_Y.shape)
 print()

 print("shape of test data ", test_X.shape)
 print("shape of test labels ", test_Y.shape)

 return train_X,train_Y,dev_X,dev_Y,test_X,test_Y 
