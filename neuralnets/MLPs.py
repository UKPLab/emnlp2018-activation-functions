from sklearn.metrics import f1_score

import os,sys
import random
import time

#sys.path.append("/work/scratch/se55gyhe/Act_func/neuralnets")

import keras
import numpy as np
from keras import Input
from keras import backend as K
from keras import callbacks
from keras.layers import Dense, Activation, Dropout, PReLU, LeakyReLU, MaxoutDense, BatchNormalization
from keras.models import Sequential
#from activation_functions import cube
#from keras.utils.generic_utils import get_custom_objects

#get_custom_objects().update({'cube': Activation(cube)})

def MLPbasic(train_data, train_labels, dev_data, dev_labels, test_data, test_labels, dataset, fun, optimizer,
             layer, dropout_value, unit, initializer, index, learning_rate,callbacks_metric = 'loss',maxout_k=3,datasize=""):
    input_dim = train_data.shape[1]
    # Number of classes
    output_dim = train_labels.shape[1]
    # Names of files where accuracy and f1 scores are saved
    myname = "__".join([fun,optimizer[0],str(layer),str(dropout_value),str(unit),str(initializer[0]),str(learning_rate),datasize])
    if fun in ["maxout","leakyrelu"]: myname = myname+"__"+str(maxout_k)
    acc_name = '%s-basic-acc%s.csv'%(myname,index)
    f1_name = '%s-basic-f1%s.csv'%(myname,index)

    # Name of directory to store results
    results_dir = dataset + '-results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_dir = results_dir+"/"+fun
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Name of directory to store models
    models_dir = dataset + '-models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    acc = open(results_dir + '/' + acc_name, 'w')
    f1_file = open(results_dir + '/' + f1_name, 'w')

    # record parameter and write them to accuracy and f1 files
    params = dataset + '-' + optimizer[0] + '-' + 'layers: ' + str(layer) + '- dropout = ' + str(dropout_value)
    acc.write('\n____________________________\n')
    acc.write(params)
    acc.write('\n____________________________\n')
    f1_file.write('\n____________________________\n')
    f1_file.write(params)
    f1_file.write('\n____________________________\n')

    acc.write('\n____________________________\n')
    acc.write('\n function: ' + fun)
    acc.write('\n____________________________\n')
    acc.write('Units,1,2,3,4,5,max,min,average,standard deviation')

    f1_file.write('\n____________________________\n')
    f1_file.write('\n function: ' + fun)
    f1_file.write('\n____________________________\n')
    f1_file.write('Units,1,2,3,4,5,max,min,average,standard deviation')
    
    acc.write('\n' + str(unit) + ',')
    f1_file.write('\n' + str(unit) + ',')
    acc_scores = []
    dev_scores = []
    f1_scores = []
    f1_dev_scores = []
    # Repeat experiment 5 times for each parameter setting
    for i in range(5):
      keras_model = Sequential()
      if callbacks_metric == 'loss':
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0,mode='auto')
        weightsPath = models_dir + '/' + str(time.time())
        checkpoint = callbacks.ModelCheckpoint(weightsPath,monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='auto')
      elif callbacks_metric == 'f1':
        earlyStopping = callbacks.EarlyStoppingF1(patience=10, verbose=0,mode='max')
        weightsPath = models_dir + '/' + str(time.time())
        checkpoint = callbacks.ModelCheckpointF1(weightsPath, verbose=1,save_best_only=True,save_weights_only=False, mode='max')
      else:
        raise Exception('Unknown metric')

      if fun!="maxout":
        keras_model.add(Dense(units=unit, input_dim=input_dim,kernel_initializer=initializer[1]))

        if fun=="prelu":
          act = PReLU(init='zero',weights=None)
        elif fun=="leakyrelu":
          act = LeakyReLU(alpha=maxout_k)
        else:
          act = Activation(fun)
        keras_model.add(act)
      else:
        keras_model.add(MaxoutDense(output_dim=unit, nb_feature=maxout_k, input_dim=input_dim,init=initializer[1]))
      keras_model.add(Dropout(dropout_value))

      for j in range(layer-1):
        if fun!="maxout":
          keras_model.add(Dense(units=unit,kernel_initializer=initializer[1]))
          if fun=="prelu":
            act = PReLU(init='zero',weights=None)
          elif fun=="leakyrelu":
            act = LeakyReLU(alpha=0.01)
          else:
            act = Activation(fun)
          keras_model.add(act)
        else:
          keras_model.add(MaxoutDense(output_dim=unit, nb_feature=maxout_k,init=initializer[1]))    
        keras_model.add(Dropout(dropout_value))

      keras_model.add(Dense(units=output_dim))
      keras_model.add(Activation('softmax'))

      k_batchsize = 16
      k_epochs = 100

      keras_model.compile(loss='categorical_crossentropy', optimizer=optimizer[1], metrics=['accuracy'])

      history = keras_model.fit(x=train_data, y=train_labels, batch_size=k_batchsize,epochs=k_epochs,verbose=2,callbacks=[earlyStopping, checkpoint],validation_data=(dev_data, dev_labels))

      # Predictions on test data
      keras_model.load_weights(weightsPath)
      preds = keras_model.predict_classes(test_data)
      preds_dev = keras_model.predict_classes(dev_data)
      score_dev = keras_model.evaluate(dev_data,dev_labels)
      score_test = keras_model.evaluate(test_data, test_labels)
      # F1 on test data
      f1_measure = f1_score(np.argmax(test_labels, axis=1), preds, average='macro')
      f1_dev = f1_score(np.argmax(dev_labels, axis=1), preds_dev, average='macro')
      # Store scores over all iterations
      dev_scores.append(score_dev[1])
      acc_scores.append(score_test[1])
      f1_scores.append(f1_measure)
      f1_dev_scores.append(f1_dev)

      K.clear_session
      # Calculate max, min, avg, std_dev for accuracy and f1 over all iterations
    acc_max = np.max(acc_scores)
    acc_min = np.min(acc_scores)
    acc_avg = np.average(acc_scores)
    acc_std = np.std(acc_scores)

    acc_scores += [acc_max, acc_min, acc_avg, acc_std]

    acc.write(','.join(str(s) for s in acc_scores))
    acc.write("\t"+",".join(str(s) for s in dev_scores))

    acc.flush()
    os.fsync(acc)

    f1_max = np.max(f1_scores)
    f1_min = np.min(f1_scores)
    f1_avg = np.average(f1_scores)
    f1_std = np.std(f1_scores)

    f1_scores += [f1_max, f1_min, f1_avg, f1_std]

    f1_file.write(','.join(str(s) for s in f1_scores))
    f1_file.write("\t"+",".join(str(s) for s in f1_dev_scores))

    f1_file.flush()
    os.fsync(f1_file)

    acc.write('\n')
    f1_file.write('\n')
    acc.close()
    f1_file.close()

