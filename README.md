# Python-Project

![](https://github.com/victorjourne/Python-Project/blob/master/emotion.gif)


[See our notebook](notebook_common.ipynb)


## Motivation
Facial expression recognition is an important and challenging task to achieve a successful human-computer interaction system.
The American psycholigist Paul Ekman demonstrated that there exists six universal emotional labels that fit facial expressions accross all cultures. Those facial expressions are: anger, disgust, fear, happiness, sadness and surprise.
After the success of convolutional neural networks (CNNs) in image classification, these models as been extended to facial expression recognition. The main advantage of using these deep learning algorithms is that CNN learns to extract the features directly from the training database.
The aim of this project is to implement deep neural networks to learn the computer to recognize facial expressions.

## Introduction
### Database
For this project, we use the database from the <a href="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge">Kaggle Facial Expression Recognition Challenge</a>. The database consists of **35887 48-by-48-pixel grayscale images**. Each face is labeled with one of the following seven emotions: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.<br />
The database is divided into three sets:
- a **training set** of 28709 examples;
- a **public test** of 3589 examples;
- a **private test** of 3589 examples.
<br />
<br />
Contrary to Kaggle competitors who had only access to the training and the public sets, we have also access to the private set. To put ourselves under the same conditions as Kaggle competitors, we will only use the training set and the public set to train our models and use the private set to test the accuracy of our model.

### Model: Convolutional Neural Network (CNN)

In the field of computer vision such as pattern recognition, neuronal networks are getting more and more popular.
We have chosen for this project a convolutional neural network which imitates the brain's working.

![](https://github.com/victorjourne/Python-Project/blob/master/CNNArchitecture.jpg)

The network is made up with several different nature of layers. The feature extraction part gathers convolutional layers acting as local pattern matching and subsampling layers used for dimensionality reduction. This part is plugged with a final classification layer of fully connected neurons.