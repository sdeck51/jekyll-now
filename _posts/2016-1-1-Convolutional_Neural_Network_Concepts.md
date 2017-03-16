---
layout: post
title: Neural Network Theory
---

In this post I go over the basics of neural networks and convolutional neural networks.

# What are Convolutional Neural Networks?
There are a lot of concepts in Convolutional Neural Networks(CNN) that need to be covered. Here we'll go through each of them. CNNs can be thought of as two concepts, one being a neural network and the other being organized via convolutions.

## Neural Networks
Neural Networks are systems of several simple elements interconnected among each other that are trained and adjusted to some requested output for some given input.
### Input, Hidden, Output Layers

### Weights, Biases, Activation Functions

### Inference/Training

### Back Propagation
<img src="https://github.com/favicon.ico" height="48" width="48"> 

#### Loss Functions
Depends on what you want to do

### Convolution
#### Kernel Sizes, Strides

## Convolutional Neural Networks

### Convolution Layer
Convolution layers consist of learnable kernels that are convolved against the input and output feature maps.  

#### Pooling Layer
Pooling, also known as subsampling, reduces the dimensionality of the previous layers feature maps. These layers also help with overfitting by filtering out higher frequency information and retaining general low frequency information that is shared across a distribution of data, rather than specific training data. Like convolution, pooling uses a sliding window technique. The difference is in the operation of the window. There are several different types of downsampling used in pooling layers, some more popular such as max pooling.

Show image of max pooling.
<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" height="400" width="500">
### Fully Connected Layer

### Additional Stuff / Classification Layer /Full Pixel Layer?


## References
<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" width="22">
<center>{% include image.html url="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png"
</center>description="Max Pooling. [CITEHERE3]" size="300" %}
