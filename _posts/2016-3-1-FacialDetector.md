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

###Code Syntax Test
{% highlight python %}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

### Used for importing the model
import urllib.request
import tarfile
import zipfile
import sys
import os
import pickle
import random

from IPython.display import Image, display

{% endhighlight %}

{% highlight ruby %}
def createConvolutionLayer(x_input, kernel_size, features, depth):
    # createConvolutionLayer generates a convolution layer in the session graph
    # by assigning weights, biases, convolution and relu function
    #
    # x_input - output from the previous layer
    # kernel_size - size of the feature kernels
    # depth - number of feature kernels
    #
    # returns convolution layer in graph
    #
    print("conv: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, features, depth],
                             initializer = tf.contrib.layers.xavier_initializer())
    
    biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0))
    
    convolution = tf.nn.conv2d(x_input, weights, strides=[1,1,1,1], padding='SAME')
    
    added = tf.nn.bias_add(convolution, biases)
    
    return tf.nn.relu(added)

{% endhighlight %}

```python
import urllib2
def function():
    #This is a function
```


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
- Already have code working to do this. Will demonstrate the beginning training process using 1 batch steps.
