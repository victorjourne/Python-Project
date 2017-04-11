# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 23:53:25 2016

@author: KD5299
"""


import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
from skimage import exposure,transform,feature
import pandas as pd
from sklearn import preprocessing


dico_emotion = {0:'Angry', 1:'Fear',
                2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}

#import getpass
#getpass.getuser()
# Importation des donnees

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
    
    def CreateUsageSet(self,usage):
        mask = np.in1d(self.data_usage, usage)
        X = self.data_image[mask, :]
        Y = self.data_emotion[mask]
    
        if K.image_dim_ordering() == 'th':
            X = X.reshape(X.shape[0], 1,self.dim, self.dim)
        else:
            X = X.reshape(X.shape[0], self.dim, self.dim, 1)
        X = X.astype('float32')
        Y = np_utils.to_categorical(Y, self.nb_classes)
        return X,Y
    # for second method
    def lbp(self,P=24,R=8):
        data_image_lbp = self.data_image*0
        i = 0
        for image in self.data_image:
            data_image_lbp[i]  = feature.local_binary_pattern(
            image.reshape((self.dim, self.dim)),P,R,method = 'uniform').ravel()
            i+=1
        self.data_image = data_image_lbp  
        
    def HistoBlocks(self,nbBlock = 36,FeatDim = 20):
        ImgDim = self.dim
        SpatialDim= int(np.sqrt(nbBlock))
        SpatialRatio = int(ImgDim/SpatialDim)

        A= np.ones((SpatialRatio,SpatialRatio))
        block = np.bmat([[a*A+b for a in range(SpatialDim)] for b in range(0,SpatialDim**2,SpatialDim)])
        def comput_histo(image):
            H,_,_ = np.histogram2d(np.asarray(block).ravel(),image,bins=(SpatialDim**2,FeatDim))
            return H.ravel()
        return np.apply_along_axis(
                                   comput_histo,1,self.data_image)
        
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

            
    def ViewOneEmotion(self,example,ax,usage=False):
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
        if usage:
            image = self.data_image[self.data_usage==usage][example]
            emotion = self.data_emotion[self.data_usage==usage][example]

        else:
            image = self.data_image[example]
            emotion = self.data_emotion[example]
        pixels = image.reshape(self.input_shape[0:2])
        ax.imshow(pixels, cmap='gray')
        ax.set_title(dico_emotion[emotion])
        plt.axis('off')
        return ax

    def ViewSomeEmotions(self,example_list,usage=False):
        fig = plt.figure(figsize=(16,8))
        i = 1
        nrow = int(np.sqrt(len(example_list)+.25)-0.5)+1
        for example in example_list:
            ax = fig.add_subplot(nrow,nrow+1,i)
            ax = self.ViewOneEmotion(example,ax,usage)
            i = i+1 
    
    def ViewEmotionPredictions(self,usage,example_list,prediction_matrix):
        nrow = 2*(int(np.sqrt(len(example_list)+.25)-0.5)+1)
        ncol = (2*len(example_list))/nrow+1
        fig = plt.figure(figsize=(12,12))
        i = 1
        for example in example_list:
            ax = fig.add_subplot(nrow,ncol,i)
            ax = self.ViewOneEmotion(usage,example,ax)
            ax1 = fig.add_subplot(nrow,ncol,i+ncol)
            ax1.bar(range(0,self.nb_classes), prediction_matrix[example],color =colors)
            ax1.set_xticks(np.arange(0.5,6.5,1))
            ax1.set_xticklabels(dico_emotion.values(), rotation=45, fontsize=7)
            ax1.set_yticks(np.arange(0.0,1.1,0.5))
#            if i%ncol==0:
#                i = i+ncol
            i = i+1+ncol*(i%ncol==0)
        plt.tight_layout()

    # Substract the mean value of each image
    def SubstractMean(self):
        mean = self.data_image.mean(axis=1)
        self.data_image = self.data_image - mean[:, np.newaxis]

    # set the image norm to 100 and standardized each pixels accross the image    
    def Normalization(self):
        # set the image norm to 100 
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
        self.data_image = min_max_scaler.fit_transform(self.data_image)
        # standardized each pixels accross the image
        scaler = preprocessing.StandardScaler().fit(self.data_image[self.data_usage=='Training'])
        self.data_image = scaler.transform(self.data_image)

    def FlipTrain(self,usage):
        flip_image = self.data_image[self.data_usage==usage]*0
        i = 0
        for image in self.data_image[self.data_usage==usage]:
            flip_image[i] = np.fliplr(
                    image.reshape(self.input_shape[0:2])).ravel()
            i=1+i
        flip_emotion = self.data_emotion[self.data_usage==usage]
        flip_usage = self.data_usage[self.data_usage==usage]+" flip"

        self.data_image = np.concatenate(
                    (self.data_image,flip_image),axis=0)
        self.data_emotion = np.concatenate(
                    (self.data_emotion,flip_emotion),axis=0)     
        self.data_usage = np.concatenate(
                    (self.data_usage,flip_usage),axis=0)
        
 