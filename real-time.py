import cv2

import time

import os

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from skimage import exposure,transform
import pandas as pd
from sklearn import preprocessing
from EmotionClass import Data
import cv2

DATA_PATH = os.environ['EMOTION_PROJECT']
#DATA_PATH = "/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/donnees/fer2013"
#DATA_PATH = "mypath"
GIT_PATH = "C:\Users\KD5299\Python-Project"
cascPath = "C:\Users\KD5299\AppData\Local\Continuum\Anaconda2\pkgs\opencv3-3.1.0-py27_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

## load data to build the scaler
#df0 = pd.read_csv(os.path.join(DATA_PATH,'fer2013.csv'), 
#                     sep=",")
#df0.drop('pixels',axis = 1,inplace=True)
#df1 = pd.read_csv(os.path.join(DATA_PATH,'pixels.csv'), 
#                             sep=" ", header=None)
#df = pd.merge(df0,df1,left_index=True,right_index=True)
#data= Data(df)
#data.SubstractMean()
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
#data.zoom(2)
#data.data_image = min_max_scaler.fit_transform(data.data_image)
## standardized each pixels accross the image
#scaler = preprocessing.StandardScaler().fit(data.data_image)
## standardized each pixels accross the image


# load model
model = load_model(os.path.join(GIT_PATH,'modelflip48'))

# dico emotion
colors = ['b', 'r', 'c', 'm', 'y', 'maroon']
dico_emotion = {0:'Angry', 1:'Fear',
                2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral',-1:'?'}


class ImgRealTime(Data):
    def __init__(self,img):
        self.data_emotion = np.array([-1])
        self.data_usage = np.array(['RealTime'])
        self.data_image = img
    def Normalization(self):
        self.data_image = min_max_scaler.fit_transform(self.data_image)
        self.data_image = scaler.transform(self.data_image)



video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)

#    plt.imshow(img_gray)
    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
#        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:

        face_image_gray = img_gray[y:y+h, x:x+w]
        resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
        img = np.expand_dims(resized_img.ravel(), axis=0)
        image=ImgRealTime(img)
#        image.SubstractMean()
#        image.Normalization()
        image.EnhanceContrast()
#        image.data_image.shape
#        image.dim
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        image.ViewOneEmotion('RealTime',0,ax)
        X, _ = image.CreateUsageSet('RealTime') 
        Y = model.predict(X)
        text =  str([dico_emotion[emo]+'  :'+'%.2f' %Y[0][emo] for emo in range(0,6) ])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame,text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)
        cv2.putText(frame,dico_emotion[Y.argmax()], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
