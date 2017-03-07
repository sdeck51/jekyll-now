---
layout: post
title: Facial Feature Detector
---

In this post I go over how to make a facial feature detector. Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/7.%20FacialFeatureDetection_Tutorial/FaceDetector.ipynb).

# Intro
This tutorial is based on an excellent post by [Daniel Nouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#dropout). I've implemented his ideas into Tensorflow. We learn how to create a convolutional neural network that can detect facial features.

# Purpose/Goal
What are the goals of this tutorial?
The purpose of this tutorial is to train a model to be able to detect facial features in an image. These features include eyes, eyebrows, nose. and mouth. 

#Data
The data used can be found [here](https://www.kaggle.com/c/facial-keypoints-detection/data). Training data consists of 7049 images. There are 30 labels, represented as an x and y coordinate for 15 features of the face. **Talk about how data is accessed. Also that its all in a csv that needs to be extracted

# Process
To begin I want to start by building a simple neural network to see how it fares. Afterwards I'll implement a larger convolutional neural network and implement concepts that have shown to boost performance of networks, namely:
#### Concepts being implemented
- dropout
- data augmentation
- optimizer optimizations (momentum, learning rate)

#### 6 networks to compare
- Basic NN
- Conv
- Conv improved learning rate
- Conv data augmentation
- Conv decaying learning rate
- Conv dropout

#### CNN Model
- 3 convolution layers
- each followed by a max pooling layer
- 3 fully connected layers
*Make picture of model
*Include dimensions

#### Progress
I have a bunch of different working models demonstrating concepts that improve the model. I would like to do additional models to improve results even more if I have enough time.

# Results
![](http://i.imgur.com/qMv2z9k.png)

Acronyms in Model Name: 
- NN - Neural Network
- CNN - Convolutional Neural Network
- LRNx - Learning Rate of x
- EX - exponential decay of learning rate
- EP - Number of Epochs trained
- DA - Data Augmentation
- DRP - Dropout enabled

Current Best Model
![](http://i.imgur.com/y1EbHby.png)

Red points represent labels, while green points represent label predictions. You can see that the third row fourth column does not match up very well. We can see that the image is both off angle and the face is somewhat tilted.

####  Additional Improvements to try: 
- Additional data augmentation: Current augmentation includes horizontal flipping, brightness, and noise. Other augmentations such as rotating the face or cropping could produce a more robust model. This additional will need labels to be transformed in the same fashion.
- separate models for specific features: Instead of having one model for all of the labels, separate models can be used to calculate specific labels, such as each eye, eyebrow, nose. and mouth. 
- continued adjustments to optimizer: using different values for learning rate and momentum have shown changes to both the speed of training as well as overall error. 

## Things I want to add
- gif showing validation image predicted labels migrate while model is being trained
