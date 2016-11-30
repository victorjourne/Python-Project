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

# Importation des donnees
#DATA_PATH = os.environ['EMOTION_PROJECT']
DATA_PATH = "/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/donnees/fer2013"
#GIT_PATH = "C:\Users\KD5299\Python-Project"
GIT_PATH = "/Users/ludoviclelievre/Documents/Python-Project"

df = pandas.read_csv(os.path.join(DATA_PATH,'fer2013.csv'), 
                     sep=",")
df.head()
data_emotion = df['emotion']
data_pixels = df['pixels']
data_usage = df['Usage']

# on exporte et on reimporte les donnees pixels pour les mettre dans un data frame
#data_pixels.to_csv(os.path.join(os.environ['EMOTION_PROJECT'],'pixels.csv'),
#                   sep="\t",encoding="utf-8", index=False)
data_image = pandas.read_csv(os.path.join(DATA_PATH,'pixels.csv'), 
                             sep=" ", header=None)

# creation de numpy arrays
data_image = data_image.as_matrix(columns=None)
data_emotion = data_emotion.as_matrix(columns=None)
data_usage = data_usage.as_matrix(columns=None) # usage = "Training", PublicTest", PrivateTest"
Xtrain = data_image[data_usage=='Training', :]
Ytrain = data_emotion[data_usage=='Training']
Xcv = data_image[data_usage=='PublicTest', :]
Ycv = data_emotion[data_usage=='PublicTest']
Xtest = data_image[data_usage=='PrivateTest', :]
Ytest = data_emotion[data_usage=='PrivateTest']

# dico emotion
dico_emotion = {0:'Angry',1:'Disgust', 2:'Fear',
                3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
# Visualize the data

def view_emotion(exemple):
    fig = plt.figure(str(exemple))
    ax = fig.add_subplot(111)
    pixels = data_image[exemple,:]
    pixels = pixels.reshape((48, 48))
    ax.imshow(pixels, cmap='gray')
    ax.set_title(dico_emotion[data_emotion[exemple]])
# test 
view_emotion(7)
# creation train set, cross validation set et test set
# on reshape les donnees
Xtrain = Xtrain.reshape((Xtrain.shape[0],48,48))
Xcv = Xcv.reshape((Xcv.shape[0],48,48))
Xtest = Xtest.reshape((Xtest.shape[0],48,48))


### Debut CNN ###

batch_size = 128
nb_classes = 7
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


# Reschape the data to insert into keras
if K.image_dim_ordering() == 'th':
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, img_rows, img_cols)
    Xcv = Xcv.reshape(Xcv.shape[0], 1, img_rows, img_cols)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    Xtrain = Xtrain.reshape(Xtrain.shape[0], img_rows, img_cols, 1)
    Xcv = Xcv.reshape(Xcv.shape[0], img_rows, img_cols, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

Xtrain = Xtrain.astype('float32')
Xcv = Xcv.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain /= 255
Xcv /= 255
Xtest /= 255

# convert class vectors to binary class matrices
YtrainBin = np_utils.to_categorical(Ytrain, nb_classes)
YcvBin = np_utils.to_categorical(Ycv, nb_classes)
YtestBin = np_utils.to_categorical(Ytest, nb_classes)

# CNN model

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
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
ax.set_xticks(np.arange(7)+0.5)
ax.set_xticklabels(tuple(dico_emotion.values()))

# save the model

model.save(os.path.join(GIT_PATH,'model1'))

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