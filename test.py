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
dico_emotion = {0:'Angry', 1:'Fear',
                2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}
                
colors = ['b', 'r', 'c', 'm', 'y', 'maroon']

# test on the whole dataset
P=24
R=2

example = Data(df.sample(10))
example.ViewSomeEmotions(range(10))
example.lbp(P,R)
example.ViewSomeEmotions(range(10))
myHist = example.HistoBlocks(nbBlock =36,FeatDim = 20) # nbBlock MUST be a perfect square, ie 4,9...

data= Data(df[df['Usage']=='Training'])
data.lbp(P,R)
data.ViewSomeEmotions(range(20))

data_test= Data(df[df['Usage']=='PrivateTest'])
data_test.lbp(P,R)
plt.hist(data_test.data_image[1,:])
# create features
histos_train =data.HistoBlocks(SpatialDim = 12,FeatDim = P/2) # saptial dim: dimension of a block, FeatDim: dimension to code lmb features
histos_test =data_test.HistoBlocks(SpatialDim =12,FeatDim = P/2) # saptial dim: dimension of a block, FeatDim: dimension to code lmb features
# label
y_train = data.data_emotion
y_test = data_test.data_emotion
histos_test.shape
nbBlock = data.dim**2/SpatialDim**2
plt.bar(np.arange(nbBlock*P),histos_test[1,:])
import xgboost as xgb
y_test.shape
histos_test.shape
# label need to be 0 to num_class -1
histos_train.shape
xg_train = xgb.DMatrix( histos_train, label=y_train)
xg_test = xgb.DMatrix(histos_test, label=y_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 7
param['subsample '] = 0.7
param['colsample_bytree '] = 0.6
 
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6




watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 150
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict( xg_test );

print ('predicting, classification error=%f' % (sum( int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
# for proba..
## do the same thing again, but output probabilities
#param['objective'] = 'multi:softprob'
#bst = xgb.train(param, xg_train, num_round, watchlist );
## Note: this convention has been changed since xgboost-unity
## get prediction, this is in 1D array, need reshape to (ndata, nclass)
#yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 6 )
#ylabel = np.argmax(yprob, axis=1)
#
#print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save('features.npy',histos_test)
np.save('labels.npy',y_test)
