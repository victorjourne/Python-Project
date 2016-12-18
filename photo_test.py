#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:48:50 2016

@author: ludoviclelievre
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

path = '/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/photo/photo_test.jpeg'
img = cv2.imread(path)
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
#image_gray.astype('float32')


face_cascade = cv2.CascadeClassifier('/Users/ludoviclelievre/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
#face = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    gray = gray[y:y+h, x:x+w]
    #res = cv2.resize(img,dsize=(48,48), interpolation = cv2.INTER_AREA)

plt.imshow(gray, cmap='gray')
res = cv2.resize(gray, (24, 24)) 
plt.imshow(res, cmap='gray')

### predict class
model_path='/Users/ludoviclelievre/Dropbox/Docs Share/projet_python'
model24 = load_model(os.path.join(model_path,'model24'))
res = res.reshape((1,24,24,1))
img_pred = model24.predict_classes(res, batch_size=128, verbose=0)






