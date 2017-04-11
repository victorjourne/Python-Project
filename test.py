# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:46:15 2017

@author: KD5299
"""
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
data.lbp(P=12,R=3)
data.ViewSomeEmotions(range(20))

histos=data.HistoBlocks(SpatialDim = 8,FeatDim = 10) # saptial dim: dimension of a block, FeatDim: dimension to code lmb features
histos.shape
plt.bar(np.arange(360),histos[1,:])

data.data_image[0].reshape((48,48))[0:8,0:8]
np.histogramdd(np.indices((48, 2))[0,:,:].ravel(),np.indices((2, 2))[1,:,:].ravel(),data.data_image[0])
idx = np.indices((48, 48))
np.indices((2, 2))[0,:,:]

SpatialDim = 8
ImgDim = 48
SpatialRatio = (ImgDim/SpatialDim)
A= np.ones((SpatialDim,SpatialDim))
block = np.bmat([[a*A+b for a in range(ImgDim/SpatialDim)] for b in range(0,SpatialRatio**2,SpatialRatio)])
block.ravel()
plt.imshow(block,interpolation='none')
plt.hist(data.data_image[0])
FeatDim = 48
H,_,_ = np.histogram2d(np.asarray(block).ravel(),data.data_image[0],bins=(SpatialRatio**2,FeatDim))
H.shape
for i in range(SpatialRatio**2):
    plt.bar(np.arange(FeatDim)+i*ImgDim,H[i,:])
