#!/usr/bin/env python

#**************
# IMPORTS
#**************
import cv2
import pickle
import sys
import os

import numpy as np
import pandas as pd

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras import models
from keras import layers

from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from math import ceil,floor
from random import choice

import matplotlib.pyplot as plt
import seaborn as sns


#**************
# INTRO
#**************

# Image size
#
SIDE = 40
# DIRECTORY FOR THE FILES!
#
# Please create the "nn-images" directory
# and populate it with the base-images
#
# MACRODIR ---.__ nn-images/100-images.png go here!
#             |
#             |__ model     ..............
#             |                          | 
#             |__ training_curve.png     |
#             |                          |--> this will be created
#             |__ small-test.pkl         |    by the script.
#             |                          |
#             |__ results.pkl            |
#             |                          |    this last one is optional.
#             |__ whole-database.pkl ....| <--It's not necessary as it's ~3Gb  
#                                             
#
MACRODIR = 'cipolina-results'


#**************
# PREPROCESSING
#**************

# FIRST SCRIPT:
#
# Transform the ~100 base-images into a rotated and contracted dataset
#
def first_one(N=30000):
  firstdir = os.getcwd()
  os.chdir(MACRODIR + '/nn-images')
  images = {name:cv2.imread(name,0) for name in os.listdir() if name[-3:] in ['png','jpg']}
  images_names = list(images.keys())
  os.chdir(firstdir)
  print(images_names)
  data = {'angle':[], 'scaling':[],
          'name':[],  'image':[],'source':[]}
  
  length = int((N/len(images_names))**(1/2))
  angles = np.linspace(-90,90,length)
  scalings = [round(2**x,3) for x in np.linspace(-1,1,length)]
  combinations = []
  lap = 0
  for _1 in images_names:
    for _2 in range(length):
      for _3 in range(length):
        combinations.append((angles[_2], scalings[_3], _1))
  for x in combinations:
    lap+=1
    if lap%10000==0: print('lap number is ',lap)

    angle = x[0]
    scaling = x[1]
    img_name = x[2]

    img = images[img_name]
    rows,cols = img.shape
    result = first_one_aux(img, rows, cols, angle, scaling)
    data['angle'].append(angle)
    data['source'].append(img_name)
    data['scaling'].append(scaling)
    data['image'].append(result)
    name_index=0
    name = f'{angle}_{scaling}_{name_index}'
    while name in data['name']:
      name_index+=1
      name = f'{angle}_{scaling}_{name_index}'
    data['name'].append(name)
  return data
    

# AUX for the FIRST SCRIPT
#
def first_one_aux(img, rows, cols, angle, scaling):
  """
  OpenCV rotates and expands/contracts 
  by angle and scaling
  """

  # ROTATE by "angle"
  M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
  new = cv2.warpAffine(img,M,(cols,rows))

  # RESCALE by "scaling"
  #
  # shrinking -->  cv2.INTER_AREA 
  # zooming   -->  cv2.INTER_CUBIC (slow) OR cv2.INTER_LINEAR (fast)
  #
  slow = False
  if slow: zoom_inter = cv2.INTER_CUBIC
  else: zoom_inter = cv2.INTER_LINEAR
  if scaling<1:
    new = cv2.resize(new,(0,0),fx=scaling, fy=scaling, interpolation = cv2.INTER_AREA)
    temp_rows, temp_cols = new.shape
    if (rows-temp_rows)%2==0:
      top,bottom = [(rows-temp_rows)//2]*2
    else:
      top,bottom = floor((rows-temp_rows)/2), ceil((rows-temp_rows)/2) 
    if (cols-temp_cols)%2==0:
      left,right = [(cols-temp_cols)//2]*2
    else:
      left,right = floor((cols-temp_cols)/2), ceil((cols-temp_cols)/2) 
    new = cv2.copyMakeBorder(new,top, bottom, left, right,  cv2.BORDER_CONSTANT, value=0)
  elif scaling>1:
    new = cv2.resize(new,(0,0),fx=scaling, fy=scaling, interpolation = zoom_inter)
    temp_rows, temp_cols = new.shape
    new = new[(temp_rows-rows)//2:(temp_rows+rows)//2, (temp_cols-cols)//2:(temp_cols+cols)//2]
    new = cv2.resize(new,(rows,cols), interpolation = cv2.INTER_AREA)

  return new


# SECOND SCRIPT
# 
# Split the ~100k database of rotated and expnaded/contracted images
# into a training, testing and validating dataset.
#
def second_one(data):
  from sklearn.model_selection import train_test_split
  L = len(data['angle'])
  for x in ['angle','scaling']: data[x] = np.asarray(data[x]).reshape(-1,1)
  X,y = data['image'], np.concatenate([data['angle'], data['scaling']],1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=420)

  data = {'train':(X_train, y_train),
          'val':(X_val, y_val),
          'test':(X_test, y_test),}

  return data


# THIRD SCRIPT
# 
# Rescale (to 0-1) and extract a tiny sample from the
# testing set to visualize afterwards
#
def third_one(data, **kwargs):
    s = MinMaxScaler().fit(data['train'][1])
    if True:
      result = {}
      def op_1(q):
        return q/255

      for x in data.keys():
        print(data[x][1].shape)
        if True:
          print(f'Y pair for key {x} max,min were: \n{np.max(data[x][1],0)}'\
                f', {np.min(data[x][1],0)}\n{np.max(data[x][1],0)}, '\
                f'{np.min(data[x][1],0)}')
        if c==1: 
          print('c==1 has a speed improvement!')
          L,*_ = data[x][0].shape
          result[x] = (data[x][0].reshape(L,SIDE,SIDE,1)/255, s.transform(data[x][1]))
        else:
          result[x] = (np.asarray([op_1(y) for y in data[x][0]]) , 
                       s.transform(data[x][1]))
        if True:
          print(f'Y pair for key {x} max,min are: \n{np.max(result[x][1],0)}'\
                f', {np.min(result[x][1],0)}\n{np.max(result[x][1],0)}, '\
                f'{np.min(result[x][1],0)}')
      data['scaler'] = (s)

      small = {'test':(result['test'][0][:150], result['test'][1][:150])}
      with open(MACRODIR + '/small-test.pkl','wb') as f: pickle.dump(small,f)
      if False:
        # we could save the entire database... 
        # or at least if it's smaller than 2.5 Gigabytes?
        if sys.getsizeof(result)<2.5e9: 
          print('SAVING THE DATABASE!')
          with open(MACRODIR + '/whole-database.pkl','wb') as f: pickle.dump(result,f)      
        else: print('database size was (in bytes)', sys.getsizeof(result))

      return result



#**************
# AI-SECTION
#**************

# The Neural Network core
def create_and_predict(data,**kwargs):
    """
    kwargs: 
        neurons=32
        epochs=50
        learning_rate=0.01
        batch_size=32
        plot=False
    """

    print('about to build network')

    #
    # 1) Initialize
 
    # This is a variation over the VGG-Network!
    # (also similar to the one used in Simone's paper about de-warping)
    
    if True:
      input_shape = (40,40,1)        # Original VGG would be (224,224,3)
      dense_neurons = 100            # Original VGG would be 4096=64^2
      f_s,f_m,f_l,f_xl = 4,8,16,32   # Original VGG would be 64, 128,256, 512
      ks_big = (3,3)                 # Original VGG would be (3,3)
      ks_small = (2,2)
      model = Sequential()
      model.add(Conv2D(input_shape=input_shape,
                      filters=f_s,kernel_size=ks_big,padding="same", activation="relu"))
      model.add(Conv2D(filters=f_s,kernel_size=ks_big,padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2,2),strides=ks_small))
      #model.add(Conv2D(filters=f_m, kernel_size=ks_big, padding="same", activation="relu")) # MUTTING 1 / 3rd of the convolutions
      model.add(Conv2D(filters=f_m, kernel_size=ks_big, padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2,2),strides=ks_small))
      #model.add(Conv2D(filters=f_l, kernel_size=ks_big, padding="same", activation="relu")) # MUTTING 1 / 3rd of the convolutions
      model.add(Conv2D(filters=f_l, kernel_size=ks_big, padding="same", activation="relu"))
      model.add(Conv2D(filters=f_l, kernel_size=ks_big, padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2,2),strides=ks_small))
      #model.add(Conv2D(filters=f_xl, kernel_size=ks_big, padding="same", activation="relu")) # MUTTING 1 / 3rd of the convolutions
      model.add(Conv2D(filters=f_xl, kernel_size=ks_big, padding="same", activation="relu"))
      model.add(Conv2D(filters=f_xl, kernel_size=ks_big, padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2,2),strides=ks_small))
      #model.add(Conv2D(filters=f_xl, kernel_size=ks_big, padding="same", activation="relu")) # MUTTING the last layer because 
      #model.add(Conv2D(filters=f_xl, kernel_size=ks_big, padding="same", activation="relu")) # our version is smaller and there's
      #model.add(Conv2D(filters=f_xl, kernel_size=ks_big, padding="same", activation="relu")) # no more room!
      #model.add(MaxPool2D(pool_size=(2,2),strides=ks_small))

      # This is the "last part" (brain?)
      #
      model.add(Flatten())
      model.add(Dense(units=dense_neurons,activation="relu"))
      model.add(Dense(units=dense_neurons,activation="relu"))
      model.add(Dense(
              2, # 2 outputs --> (angle, scaler)
              activation='linear'))

    # 2) Compile & Fit
    model.compile(
              optimizer=SGD(learning_rate=kwargs.get('learning_rate',.02)),
              loss='mean_squared_error',
              metrics='accuracy',)     
    print(model.summary())
    print(f'training set is shaped: {data["train"][0].shape} and the first dim is the # of samples')
    print('about to train')
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',4096),
            epochs=kwargs.get('epochs',50),
            verbose=1,
            validation_data=data['val'],)
    model.save(MACRODIR + '/model')

    # 3) Return results
    results = results.history 
    results['ytrue_test'] = data['test'][1]
    results['ypred_test'] = model.predict(data['test'][0])
    important_results = {'ytrue_test':results['ytrue_test'],
                         'ypred_test':results['ypred_test']}
    with open(MACRODIR + '/results.pkl','wb') as w:
      pickle.dump(important_results,w)
 
    # 4) Plot 
    plt.plot(results['accuracy'],c='b',label='train')
    plt.plot(results['val_accuracy'],c='g')
    plt.plot(results['loss'],c='b')
    plt.plot(results['val_loss'],c='g',label='validation')
    plt.legend()
    plt.savefig(MACRODIR + '/training_curve.png')

    return results


if __name__=='__main__':
    if True:
      print('about to preprocess')
      N_samples = 140000
      dat = third_one(second_one(first_one(N_samples)), mode='CNN')
      print('ended preprocessing!')
      
      print('about to train a neural network!')
      create_and_predict(dat,
              epochs=120, plot=True, learning_rate=0.03, mode='CNN')
      print('ended training a neural network!')


