---
layout: post
title: Neural Network Theory
---

In this post I go over the basics of neural networks and convolutional neural networks.

# What are Convolutional Neural Networks?
There are a lot of concepts in Convolutional Neural Networks(CNN) that need to be covered. Here we'll go through each of them. CNNs can be thought of as two concepts, one being a neural network and the other being organized via convolutions.

## Neural Networks
Neural Networks are systems of several simple elements interconnected among each other that are trained and adjusted to some requested output for some given input. 

### Weights, Biases, Activation Functions
Below is a simple neural network, that has 2 inputs, an activate function, and an output. Between the connections are weights, and each activation function holds a constant bias.

*IMAGE HERE SHOWING BASIC PERCEPTRON*

The mathematical model for the neural network above is

output = sgn(sum(weight * input + bias))

where sgn(s) 1 if s positive

This network is called the perceptron. It can model linear regression or classification by modifying the weights to change the slope of the equation, or bias for shifting the equation. Modern convolutional neural networks still follow these basic ideas of weights and biases. Nonlinear problems however need nonlinear activation functions.

#### Activation Function
The purpose of the activation function is to add nonlinearity to a model. The function sgn(s) simply turns on or off the input that is being assigned. Modern networks do this as well, though with nonlinear functions. Popular activation functions are sigmoid, tanh, and linear rectifiers.

### Inference/Training


<center>{% include image.html url="http://i.imgur.com/8QlqiS4.png"
</center>description="Simple example showing what Learning. [CITEHERE3]" size="300" %}
When training a model you have two pieces of data. The actual data, and the label for the data. When you put a piece of data into your model, you retrieve a predicted label. With the actual and predicted label you can quantify the error. This is done using a cost function, which can be specified .

#### Cost Functions
The cost function is used to determine the error between your model's predicted label and the actual label. There are many different cost functions used for different applications.


### Back Propagation




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
