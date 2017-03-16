---
layout: post
title: Tensorflow - Facial Feature Detector
---
![](http://i.imgur.com/90KjE6A.png?2)

In this post I go over how to make a facial feature detector. Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/7.%20FacialFeatureDetection_Tutorial/FaceDetector.ipynb).

# Intro


# Purpose/Goal
The main goal for this tutorial is to demonstrate how one can build a facial feature detector from scratch using tensorflow. We'll go through several different models to demonstrate how one can make improvements that lead to an optimized model. We'll also talk about improvements that can be made.

# Data
The data for a feature detector is fairly important. Unlike classification where you can simply define a label to an image, feature detection needs to know where the features in the image are. The data I'm using can be found [here](https://www.kaggle.com/c/facial-keypoints-detection/data). Labeled data consists of 7049 images. There are 30 unique labels, represented as an x and y coordinate for 15 features of the face. 
![](http://i.imgur.com/rPjZh9h.png)
<p align="center">
  <http://i.imgur.com/rPjZh9h.png />
</p>
The data is formatted in a csv file, where each row represents an image and it's labels. 
**Talk about how data is accessed. Also that its all in a csv that needs to be extracted

# Process
To begin I want to start by building a simple neural network to see how it fares. Afterwards I'll implement a larger convolutional neural network and implement concepts that have shown to boost performance of networks, namely:

To start off we're going to look at a very simple neural network and see what kind of results we can obtain from that. To do this we need to be able to build fully connected layers.

{% highlight python %}
def createFullyConnectedLayer(x_input, width):
    # createFullyConnectedLayer generates a fully connected layer in the session graph
    # 
    # x_input - output from previous layer
    # width - width of the layer (eg for a 10 class output you need to end with a 10 width layer
    #
    # returns fully connected layer in graph
    #
    print("fc: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[x_input.get_shape()[1], width],
                             initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[width], initializer=tf.constant_initializer(0))
     
    matrix_multiply = tf.matmul(x_input, weights)
    
    return tf.nn.bias_add(matrix_multiply, biases)
{% endhighlight %}

We'll need to be making multiple fully connected layers, as well as other types of layers, so I'll be making functions to generate these. The above will create a fully connected layer. Along with this we need to implement the node function. In neural networks each node has some sort of saturating function, such as the sigmoind, or hyperbolic tangent. For deep networks, due to speed concerns as well as vanishing gradient issues, linear rectifiers are used. Also called ReLu, this function is simply the function x with a floor of zero.

{% highlight python %}
def createLinearRectifier(x_input):
    # createLinearRectifier generates a ReLu in the session graph
    # 
    # The reason this exists is due to the last fully connected layer not needing a relu while others do
    # x_input - output from previous layer
    # width - width of the layer
    #
    # returns ReLu in graph
    # 
    
    return tf.nn.relu(x_input)
{% endhighlight %}

These two functions are all we need for now to build a simple model. We can create one more function that will be used to actually build the model.

{% highlight python %}
def createNetwork1(x_input):
    with tf.variable_scope('in'):
        x_input = tf.reshape(x_input, [-1, image_size*image_size])
    with tf.variable_scope('hidden'):
        hidden_fully_connected_layer = createFullyConnectedLayer(x_input, 100)
        relu_layer = createLinearRectifier(hidden_fully_connected_layer)
    with tf.variable_scope('out'):
        return createFullyConnectedLayer(relu_layer, 30)
{% endhighlight %}


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
