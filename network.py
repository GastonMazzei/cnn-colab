#!/usr/bin/env python
# coding: utf-8

"""Neural Network fits the data and saves trained-models
"""

import pickle
import sys
import os

import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras import models
from keras import layers

from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler as ss, MinMaxScaler
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

SIDE = 40

def load():
    with open('./database_processed_first.pkl','rb') as w:
        data = pickle.load(w)
    return data



def preprocess(data, **kwargs):
    s = MinMaxScaler().fit(data['train'][1])
    if False: pass
    else:
      result = {}
      try: a,b,c = data['test'][0][0].shape
      except: c=0 
      if c>1:
        def op_1(q):
          q=np.mean(q,2).reshape(SIDE,SIDE,1)        
          #return np.rint(q/255)
          return q/255
      elif c==1:
        def op_1(q):
          #return np.rint(q/255)
          return q/255
      else:
        def op_1(q):
          #return np.rint(q/255).reshape(SIDE,SIDE,1)
          return q.reshape(SIDE,SIDE,1)/255

      for x in data.keys():
        print(data[x][1].shape)
        if True:
          print(f'Y pair for key {x} max,min were: \n{np.max(data[x][1],0)}'\
                f', {np.min(data[x][1],0)}\n{np.max(data[x][1],0)}, '\
                f'{np.min(data[x][1],0)}')
        result[x] = (np.asarray([op_1(y) for y in data[x][0]]) , 
                     s.transform(data[x][1]))
        if True:
          print(f'Y pair for key {x} max,min are: \n{np.max(result[x][1],0)}'\
                f', {np.min(result[x][1],0)}\n{np.max(result[x][1],0)}, '\
                f'{np.min(result[x][1],0)}')
      data['scaler'] = (s)
      with open('./processed_database_cnn.pkl','wb') as w:
        pickle.dump(result, w)


def create_and_predict(data,**kwargs):
    """
    kwargs: 
        neurons=32
        epochs=50
        learning_rate=0.01
        batch_size=32
        plot=False
    """
    #
    # 1) Initialize

    if True:
      def L0(ks=8,f=1,s=1, act=None, pd='valid'): 
        return  layers.Conv2D(
                          f, #filters
                          (ks,ks), #kernel size
                          strides=(s, s),
                          activation=act,
                          padding = pd,
                          input_shape=(SIDE, SIDE, 1),
                          )

      def L(ks=8,f=1,s=1, act=None, pd='same'):
        return  layers.Conv2D(
                          f, 
                          (ks,ks), 
                          strides=(s, s),
                          activation=act,
                          padding=pd,
                          )
      def MP0(ps=6, s=2):
        return  layers.MaxPooling2D(pool_size=(ps, ps), strides=s, 
                                )
      def MP():
        return  layers.MaxPooling2D(pool_size=(5, 5), strides=6, 
                                )

      TIMES = 3
      out=SIDE*SIDE

      model = models.Sequential()

      N = 5
      Ndense=3
      activ = 'tanh'
      kernel_size = 30#5
      q = 30

      if True:
        # CONV 1
        model.add(L0(ks=5,f=1,s=1, act=None, pd='valid')) # dof= ks**2+1
        # POOL 1
        model.add(MP0(ps=5,s=2))
        # CONV 2-N
        for _ in range(N):
          model.add(L0(ks=kernel_size, f=1, s=1, act=activ, pd='same')) # dof= ks**2+1
        # POOL 2
        model.add(MP0(ps=5,s=2))

        # Flatten & Dense
        model.add(layers.Flatten()) 
        for _ in range(Ndense):
          model.add(Dense(
                q,#12,
                activation=activ))

      mode = ['classifier', 'regressor'][1]
      if mode=='classifier':  
        model.add(Dense(
                out,
                activation='sigmoid'))

        model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.001)),
                loss='binary_crossentropy',
                metrics='accuracy',)     
      else:  
        model.add(Dense(
                2,
                activation='linear'))

        model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.02)),
                loss='mean_squared_error',
                metrics='accuracy',)     

    #
    # 2) Fit
    print(model.summary())
    print(f'training set is shaped: {data["train"][0].shape} and the first dim is the # of samples')
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',2048),
            epochs=kwargs.get('epochs',50),
            verbose=1,
            validation_data=data['val'],)
    model.save(f'./model')

    #
    # 3) Return results
    results = results.history 
    results['ytrue_val'] = data['val'][1]
    results['ytrue_test'] = data['test'][1]
    results['ypred_val'] = model.predict(data['val'][0])
    results['ypred_test'] = model.predict(data['test'][0])
    results['specs'] = kwargs
    with open('results.pkl','wb') as w:
      pickle.dump(results,w)
 
    #
    # 4) Maybe, plot
    if kwargs.get('plot',False):
        regression = True
        case = 'val'
        if not regression:
          f, ax = plt.subplots(1,3)
          fpr, tpr, treshold = roc_curve(
                results['ytrue_'+case], results['ypred_'+case]
                    )
          ax[0].plot(fpr, tpr)
        
          weights = {0:[],1:[]}
          for i,x in enumerate(results['ypred_'+case]):
            weights[data[case][1][i][0]] += [x[0]]

       
          ax[1].hist(weights[0],label='0',alpha=0.5)
          ax[1].hist(weights[1],label='1',alpha=0.5)
          ax[1].set_xlim(0,1)
          ax[1].legend()

          ax[2].plot(results['accuracy'],c='b',label='train')
          ax[2].plot(results['val_accuracy'],c='g')
          ax[2].plot(results['loss'],c='b')
          ax[2].plot(results['val_loss'],c='g',label='validation')
          ax[2].legend()
          ax[2].set_ylim(0,1)

        else:
          plt.plot(results['accuracy'],c='b',label='train')
          plt.plot(results['val_accuracy'],c='g')
          plt.plot(results['loss'],c='b')
          plt.plot(results['val_loss'],c='g',label='validation')
          plt.legend()

        plt.show()
        if False:
            plt.plot(
                *roc_curve(
                    results['ytrue_test'], results['ypred_test']
                        )[:-1])
    return results


def build_database():
  from sklearn.model_selection import train_test_split
  with open('./database.pkl','rb') as w:
   data = pickle.load(w)  
  L = len(data['angle'])
  for x in ['angle','scaling']: data[x] = np.asarray(data[x]).reshape(-1,1)
  X,y = data['image'], np.concatenate([data['angle'], data['scaling']],1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=420)

  data = {'train':(X_train, y_train),
          'val':(X_val, y_val),
          'test':(X_test, y_test),}
  with open('./database_processed_first.pkl','wb') as w:
   pickle.dump(data, w)
  

    
if __name__=='__main__':
   
    mode = 'cnn'
    switch = {1:[True, False][0],
              2:[True, False][0],
              3:[True, False][0],}


    if switch[1]:
      # build database
      build_database()
   
    if switch[2]:
      # process database 
      preprocess(load(), mode=mode.upper())

    if switch[3]:
      with open(f'./processed_database_{mode}.pkl','rb') as f:
        dat = pickle.load(f)
      create_and_predict(dat,
              neurons=int(sys.argv[1]), epochs=int(sys.argv[2]),plot=True, mode=mode.upper())


