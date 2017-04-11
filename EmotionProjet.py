#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:16:40 2016

@author: ludoviclelievre
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
import os
from skimage import exposure,transform,feature
import pandas as pd
from sklearn import preprocessing
import pickle
# import Class Data
from EmotionClass import Data


#import getpass
#getpass.getuser()
# Importation des donnees

np.random.seed(1337)  # for reproducibility

DATA_PATH = os.environ['EMOTION_PROJECT']
#DATA_PATH = "/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/donnees/fer2013"
#DATA_PATH = "mypath"
GIT_PATH = "C:\Users\KD5299\Python-Project"
#GIT_PATH = "/Users/ludoviclelievre/Documents/Python-Project"
df0 = pandas.read_csv(os.path.join(DATA_PATH,'fer2013.csv'), 
                     sep=",")
df0.drop('pixels',axis = 1,inplace=True)
df1 = pandas.read_csv(os.path.join(DATA_PATH,'pixels.csv'), 
                             sep=" ", header=None)

df = pd.merge(df0,df1,left_index=True,right_index=True)
# drop disgut 
df = df[df['emotion']!=1]
df['emotion'] = (df['emotion']-1).where(df['emotion']>1,other = df['emotion'])
# dico emotion
dico_emotion = {0:'Angry', 2:'Fear',
                3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
                
colors = ['b', 'r', 'c', 'm', 'y', 'maroon']

# test on the whole dataset  
data= Data(df.ix[0:20])
data.lbp(P=16,R=8)
data.ViewSomeEmotions(range(20))
# see one image
fig = plt.figure()
ax = fig.add_subplot(111)
ax = data.ViewOneEmotion(10,ax)

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
one_image.EnhanceContrast()
one_image.ViewEmotion()

one_image = Data(df.iloc[3:4])
image = one_image.data_image
np.fliplr(image)

some_images = Data(df[df['Usage']=='Training'].sample(8))
print('The subset is shaped as {}'.format(some_images.input_shape))
print('The subset has {} examples'.format(some_images.nb_example))
print('The subset has {} different classes'.format(some_images.nb_classes))
# see the 8th first images of the training set
some_images.ViewSomeEmotions(range(8))

some_images.FlipTrain()

#==============================================================================
# CNN
#==============================================================================
#new dico without the under represented class digust

dico_emotion = {0:'Angry', 1:'Fear',
                2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}

data = Data(df[df['emotion']!=1])
f = lambda x: x-1 if x>1 else x
fv = np.vectorize(f)
data.data_emotion = fv(data.data_emotion)
data.FlipTrain('Training') # create 'Training flip'
data.SubstractMean()
data.Normalization()
data.zoom(2)
data.nb_example
# set inputs and outputs
Xtrain, YtrainBin = data.CreateUsageSet(['Training','Training flip']) # add 'Training flip'
Xcv, YcvBin = data.CreateUsageSet('PublicTest')
Xtest, YtestBin = data.CreateUsageSet('PrivateTest')

### parameters CNN ###
batch_size = 128
nb_epoch = 6
# input image dimensions
img_rows, img_cols = data.dim,data.dim
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

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
history24 = model.fit(Xtrain, YtrainBin, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(Xcv, YcvBin))
model.save(os.path.join(GIT_PATH,'model24'))
history24.save(os.path.join(GIT_PATH,'history24'))
model = load_model(os.path.join(GIT_PATH,'modelflip24'))
history = pickle.load(open( os.path.join(GIT_PATH,'historyflip24'), "rb" ))

# Model evaluation
# Cross validation score
scoreCV = model.evaluate(Xcv, YcvBin,verbose=0)
print('CV score:', scoreCV[0])
print('CV accuracy:', scoreCV[1])
# Test score
scoreTest = model.evaluate(Xtest, YtestBin, verbose=0)
print('Test score:', scoreTest[0])
print('Test accuracy:', scoreTest[1])

# is it relevent to flip the train set? Test of accuracy with tang model
# to compare with the model_tang_flip
data = Data(df)
data.SubstractMean()
data.TangPreprocess()
data.zoom(2)
data.FlipTrain('PublicTest')
Xcv, YcvBin = data.CreateUsageSet('PublicTest flip')
scoreTest = model.evaluate(Xcv, YcvBin , verbose=0)
print('Test score:', scoreTest[0])
print('Test accuracy:', scoreTest[1])

Xcv, YcvBin = data.CreateUsageSet('PublicTest')
scoreTest = model.evaluate(Xcv, YcvBin , verbose=0)
print('Test score:', scoreTest[0])
print('Test accuracy:', scoreTest[1])

# save the model

model.save(os.path.join(GIT_PATH,'model_tang_flip'))
model_loaded = load_model(os.path.join(GIT_PATH,'modelflip48'))
model_loaded.summary()
model_loaded.save_weights(os.path.join(GIT_PATH,'modelflip48.hdf5'))
with open(os.path.join(GIT_PATH,'modelflip48.json'), 'w') as f:
    f.write(model_loaded.to_json())



#==============================================================================
#  Results
#==============================================================================
# see some  images
PredTest = model.predict(Xtest)
data = Data(df[(df['emotion']!=1)&(df['Usage']=='PrivateTest')])
f = lambda x: x-1 if x>1 else x
fv = np.vectorize(f)
data.data_emotion = fv(data.data_emotion)

data.ViewEmotionPredictions(range(0,10),PredTest[range(0,10)])

# see some misclassed images
misclass = PredTest.argmax(axis=1)!=data.data_emotion
np.sum(misclass)
misclass_array, = np.where(misclass==True)
misclass_list = list(np.random.choice(misclass_array,20))
data.ViewEmotionPredictions(misclass_list,PredTest)

# see intermediate layer response

import theano
def plot_interlayer_outputs(input_img, layer_num1, layer_num2):
    output_fn = theano.function([model.layers[layer_num1].input], # import theano
                                 model.layers[layer_num2].output,
                                 allow_input_downcast=True)
    im = output_fn(input_img).squeeze() #filtered image
    print("shape of this layer: {}".format(im.shape))
    n_filters = im.shape[-1]
    fig = plt.figure(figsize=(12,6))
    for i in range(n_filters):
        ax = fig.add_subplot(n_filters/16,16,i+1)
        ax.imshow(im[:,:,i], cmap='gray') 
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()

example = 4
fig = plt.figure()
ax = fig.add_subplot(111)
data.ViewOneEmotion("PrivateTest",example,ax)
img = Xtest[example:example+1]
plot_interlayer_outputs(img, 0, 0)
plot_interlayer_outputs(img, 0, 1)
plot_interlayer_outputs(img, 0, 2)
plot_interlayer_outputs(img, 0, 3)
plot_interlayer_outputs(img, 0, 4)

ax = fig.add_subplot(n_filters/16,16,i+1)



