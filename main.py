#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:30:57 2018

@author: xor
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:46:21 2018

@author: xor
"""
from sklearn.neighbors import KNeighborsClassifier  
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle 
import os
import keras
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from scipy import ndimage, misc
train_images=[]
labels=[]
test_x=[]
test_y=[]
def plot(data,meta,index):
    #select the image
    image=data[b'data'][index, :]
    #collect their rgb prop and use them to plot the image
    image_red=image[0:1024].reshape(32,32)
    image_green=image[1024:2048].reshape(32,32)
    image_blue=image[2048:].reshape(32,32)
    fimage=np.dstack((image_red,image_green,image_blue))
    #`print("shape",fimage.shape)
    #print("label:", data[b'labels'][index])
    #print("category",meta[b'label_names'][data[b'labels'][index]])
    plt.imshow(fimage)
    return fimage
def label(data,meta,index):
    #select the image
    image=data[b'data'][index, :]
    #collect their rgb prop and use them to plot the image
    image_red=image[0:1024].reshape(32,32)
    image_green=image[1024:2048].reshape(32,32)
    image_blue=image[2048:].reshape(32,32)
    fimage=np.dstack((image_red,image_green,image_blue))
    #print("shape",fimage.shape)
    #print("label:", data[b'labels'][index])
    #print("category",meta[b'label_names'][data[b'labels'][index]])
    #plt.imshow(fimage)
    return meta[b'label_names'][data[b'labels'][index]]
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
meta = unpickle('/Users/xor/Downloads/cifar/batches.meta')
for x in range(0,9000,1):
   train_images.append(plot(dict,meta,x))
   labels.append(label(dict,meta,x))
for x in range(9000,9999,1):
    test_x.append(plot(dict,meta,x))
    test_y.append(label(dict,meta,x))

for x in range(0,9000,1):
    if (labels[x]==b'cat'):
        labels[x]=1
    elif (labels[x]==b'dog'):
        labels[x]=2
    elif (labels[x]==b'deer'):
        labels[x]=3
    elif (labels[x]==b'frog'):
        labels[x]=4
    elif (labels[x]==b'bird'):
        labels[x]=5
    elif (labels[x]==b'ship'):
        labels[x]=6
    elif (labels[x]==b'airplane'):
        labels[x]=7
    elif (labels[x]==b'automobile'):
        labels[x]=8
    elif (labels[x]==b'horse'):
        labels[x]=9
    elif (labels[x]==b'truck'):
        labels[x]=0
        
train_images=np.array(train_images)
labels=np.array(labels)
las = to_categorical(labels, num_classes=None)
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
def createModel():
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))

    return model
model=createModel()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
