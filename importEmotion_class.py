#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:16:40 2016

@author: ludoviclelievre
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import os
from skimage import exposure,transform
import pandas as pd
#import getpass
#getpass.getuser()
# Importation des donnees
#DATA_PATH = os.environ['EMOTION_PROJECT']
DATA_PATH = "/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/donnees/fer2013"
#DATA_PATH = "mypath"
#GIT_PATH = "C:\Users\KD5299\Python-Project"
GIT_PATH = "/Users/ludoviclelievre/Documents/Python-Project"

df0 = pandas.read_csv(os.path.join(DATA_PATH,'fer2013.csv'), 
                     sep=",")
df0.drop('pixels',axis = 1,inplace=True)
df1 = pandas.read_csv(os.path.join(DATA_PATH,'pixels.csv'), 
                             sep=" ", header=None)

df = pd.merge(df0,df1,left_index=True,right_index=True)
# dico emotion
dico_emotion = {0:'Angry',1:'Disgust', 2:'Fear',
                3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

class Data:
    def __init__(self,df):
        self.data_emotion = df['emotion'].as_matrix(columns=None)
        self.data_usage = df['Usage'].as_matrix(columns=None)
        self.data_image = df[list(filter(lambda pxl: type(pxl)!=str ,df.columns.tolist()))].as_matrix(columns=None)
   
    @property
    def nb_example(self):
        return int(self.data_emotion.shape[0])
    @property
    def dim(self):
        return int(np.sqrt(self.data_image[0].shape[0]))     
    @property
    def nb_classes(self):
        return int(np.unique(self.data_emotion).shape[0])
    @property
    def input_shape(self):
        if K.image_dim_ordering() == 'th':
            return (1, self.dim, self.dim)
        else:
            return (self.dim, self.dim,1)
 
    def zoom(self,z):
        data_image_zoom = np.ndarray((self.data_image.shape[0],
                                      self.data_image.shape[1]/z**2))
        i = 0
        for image in self.data_image:
            data_image_zoom[i] = transform.downscale_local_mean(
                            image.reshape((self.dim, self.dim)),(z,z)).ravel()
            i=1+i
        self.data_image = data_image_zoom    
#        self.dim = int(self.dim / z)
        
    def EnhanceContrast(self):
        self.data_image = np.apply_along_axis(
                                exposure.equalize_hist,1,self.data_image)
        
    def Normalize(self):
        self.data_image =self.data_image/255.

    def CreateUsageSet(self,usage):
        X = self.data_image[self.data_usage==usage, :]
        Y = self.data_emotion[self.data_usage==usage]

        if K.image_dim_ordering() == 'th':
            X = X.reshape(X.shape[0], 1,self.dim, self.dim)
        else:
            X = X.reshape(X.shape[0], self.dim, self.dim, 1)
        X = X.astype('float32')
        Y = np_utils.to_categorical(Y, self.nb_classes)
        return X,Y

    def ViewEmotion(self):
        fig = plt.figure()
        i = 1
        nrow = int(np.sqrt(self.nb_example+.25)-0.5)+1
        for emotion,image in zip(self.data_emotion,self.data_image):
            ax = fig.add_subplot(nrow,nrow+1,i)
            pixels = image.reshape(self.input_shape[0:2])
            ax.imshow(pixels, cmap='gray')
            ax.set_title(dico_emotion[emotion])
            plt.axis('off')
            i = i+1
        
# test on the whole dataset  
data= Data(df)
data.data_image[1,:]
X,Y = data.CreateUsageSet('Training')
data.EnhanceContrast()
data.dim
X[1:2,:].shape
Xe,Ye = data.CreateUsageSet('Training')
data.zoom(2)
Xez,Yze = data.CreateUsageSet('Training')
data.data_image
Xez.shape

# test with one image
one_image = Data(df.iloc[3:4])
one_image.zoom(2)
one_image.ViewEmotion()
one_image.EnhanceContrast()
one_image.ViewEmotion()

one_image = Data(df.iloc[3:4])
one_image.EnhanceContrast()
one_image.zoom(2)
one_image.ViewEmotion()

several_images = Data(df[df['emotion']==1].sample(20))
several_images.nb_example
several_images.ViewEmotion()
several_images.zoom(2)

#==============================================================================
# CNN
#==============================================================================
data = Data(df)
data.zoom(2)
data.input_shape
data.Normalize()
# set inputs and outputs
Xtrain, YtrainBin = data.CreateUsageSet('Training')
Xcv, YcvBin = data.CreateUsageSet('PrivateTest')
Xtest, YtestBin = data.CreateUsageSet('PublicTest')

### parameters CNN ###
batch_size = 128
nb_epoch = 12
# input image dimensions
img_rows, img_cols = data.dim,data.dim
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (5, 5)

# CNN model
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=data.input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(data.nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Model fitting
model.fit(Xtrain, YtrainBin, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(Xcv, YcvBin))
# Model evaluation
# Cross validation score
scoreCV = model.evaluate(Xcv, YcvBin,verbose=0)
print('CV score:', scoreCV[0])
print('CV accuracy:', scoreCV[1])
# Test score
scoreTest = model.evaluate(Xtest, YtestBin, verbose=0)
print('Test score:', scoreTest[0])
print('Test accuracy:', scoreTest[1])


predictCV = model.predict(Xcv, verbose=0)
np.sum((predictCV.argmax(axis=1)-YcvBin.argmax(axis=1))==0)/float(YcvBin.shape[0])
predictCV[0]

fig = plt.figure()
ax = fig.add_subplot(111)
j=0
for clas in range(7):
    j=j+0.1
    predClas = predictCV[YcvBin.argmax(axis=1)==clas].mean(axis=0)
    print(predClas)   
    rects = ax.bar(np.arange(7)+j, predClas.T, 0.1,
                 label='Men')
propCla  = YcvBin.mean(axis=0)
ax.set_xticks(np.arange(7)+0.5)
ax.set_xticklabels(tuple(dico_emotion.values()))

# save the model

model.save(os.path.join(GIT_PATH,'model_eq'))

model_loaded = load_model(os.path.join(GIT_PATH,'model1'))

model_loaded.summary()

# see an example and its prediction
exemple = 28716
input_image = data_image.reshape((data_image.shape[0],48,48))[exemple]
input_image = input_image.astype('float32')/255
input_image = Xtrain[1:2,:]
if K.image_dim_ordering() == 'th':
    input_image = input_image.reshape(1, 1, img_rows, img_cols)
else:
    input_image = input_image.reshape(1, img_rows, img_cols, 1)

pred = model_loaded.predict(input_image)

fig = plt.figure()
ax = fig.add_subplot(111)
rects = ax.bar(np.arange(7), pred.T, 0.35,
                 color='b',
                 label='Men')
ax.set_xticks(np.arange(7))
ax.set_xticklabels(tuple(dico_emotion.values()))

view_emotion(exemple)

# normalised histogram

img_eq = exposure.equalize_hist(Xtrain[2:3,:])

def view_emotion(exemple):
    fig = plt.figure(str(exemple))
    ax= fig.add_subplot(211)
    ax1= fig.add_subplot(221)
    pixels = data_image[exemple,:]
    pixels = pixels.reshape((48, 48))
    ax1.imshow(pixels, cmap='gray')
    ax.imshow(exposure.equalize_hist(pixels), cmap='gray')
    ax.set_title(dico_emotion[data_emotion[exemple]])

view_emotion(20)

plt.hist(img_eq.ravel(),256)
plt.hist(Xtrain[1:2,:].ravel(),256)
