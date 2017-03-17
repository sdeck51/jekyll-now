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

## Inference/Training

When training a model you have two pieces of data. The actual data, and the label for the data. When you put a piece of data into your model, you retrieve a predicted label. With the actual and predicted label you can quantify the error of the model. This is performed using a cost function.

### Cost Functions
The cost function is used to determine the error between your model's predicted label and the actual label. Take for example multi-class classification, where each input has a single class out of more than 2 classes. Each label per input could be viewed as a one hot encoded vector, say (0, 0, 0, 1), where the 1 is the class it belongs to. If a model predicts that that inputs label is (.1, .05, .15 .7), then we can use a cost function to determine the error between them. Obviously if they were equal then the error would be zero. There are many different types of cost functions, used for various applications


squared error
sum(1/2(label-prediction)
cross entropy


graphs of each here and their equations


### Backpropagation
The purpose of backpropagation is to optimize the weights of a neural network so it can learn the correct output or label, for a given input. Let's go through a small example to demonstrate what it being done.

example
in1

in2

The weights are initialized as w1 = .15, w2 = .20, w3 = .25, w4 = .3, w5 = .40, w6 = .45. 

#### Forward Pass
Backpropogation can be broken down into two phases. The first is the forward pass. The forward pass simply calculates/predicts to output of a given input. For the initialized weights lets put in an input. Let our input be i1 = .05 and i2 be .10. 

h1 = w1*i1 + w3*i2 = .15*.05 + .25*.10 = 0.0075 + 0.025 = 0.0325
h2 = w3*i1 + w4*i2 = .25*.05 + .30*.10 = 0.0125 + 0.03 = 0.0425

Assume we have sigmoid activation functions for the hidden and output layers

outh1 = sig(h1) = 0.50812
outh2 = sig(h2) = 0.51062

o = w5*outh1 + w6*outh2 = .40*0.50812 + .45*0.51062 = 0.20325 + 0.22978 = 0.43303

out = sig(o) = 0.60659

Our output is 0.60659 for the given input. Now lets say that for that given input, the attached label is 1. We can use the cost function to determine the error. Lets use a squared error cost function.

Error = 1/2(1-0.60659)^2 = 0.07738

We've calculated a forward pass and calculated the error with respect to a squared error cost function. Now we can look into the next step in backpropagation.

#### Backwards Pass
The backwards pass is where we update the weights in the network to make the predicted output closer to the label than it was. This process involves calculus. We want to look at each weight and determine how much it affects the error, or in other words we want to calculate the partial derivative with respect to some weight.

dE/dw5 = dE/dout * dout/do * do/w5 (using chain rule)

We need to calculate eachpartial derivative.

E = 1/2(label - out)^2

dE/dout = label - out = 1 - 0.60659 = 0.39341

out = sig(o)

dout/do = out(1-out) = 0.60659(1-0.60659) = .23864

o = w5*outh1 + w6*outh2

do/dw5 = outh1 = 0.50812

dE/dw5 = 0.39341 * .23864 * 0.50812 = 0.04770

new_w5 = w5 - n*dE/dw5 = 0.40 - 0.5*0.04770 = 0.37613 (where n is the user set learning rate)

This is used for every weight, going down each layer. The weights are adjusted such that their prediction is closer to the actual label. Calculation of the partial derivative can be done using several different algorithms.

#### Stochastic Gradient Descent (SGD)




## Convolutional Neural Networks
<center>{% include image.html url="http://i.imgur.com/6Xe6Nz7.png"
description="LeNET. [CITEHERE3]" size="600" %}</center>
Moving on from perceptrons and simple neural networks we get into convolutional neural networks (ConvNet). These networks, as their name describes, use convolution through how the weights are tied between layers. Each pixel in each channel of a ConvNet represents a single input, so you can imagine these networks are fairly large. Their convolution layers allow for less interconnections between layers as weights are shared rather than are unique, and the structure retains spatial information since it's convolution. Let's quickly go over what convolution is, and look at the main components used to build a ConvNet.

### Convolution
http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif
<center>{% include image.html url="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png"
description="Max Pooling. [Feature extraction using convolution, Stanford]" size="300" %}</center>
*picture* A conv B =  sumx[n]h[n-k]

#### Kernel Sizes, Strides

### Convolution Layer
Convolution layers consist of learnable kernels that are convolved against the input and output feature maps.  What this means is a layer has multiple kernels, or groups of weights, that are applied to the image. These kernels 

#### Pooling Layer
Pooling, also known as subsampling, reduces the dimensionality of the previous layers feature maps. These layers also help with overfitting by filtering out higher frequency information and retaining general low frequency information that is shared across a distribution of data, rather than specific training data. Like convolution, pooling uses a sliding window technique. The difference is in the operation of the window. There are several different types of downsampling used in pooling layers, some more popular such as max pooling.

Show image of max pooling.
<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" height="400" width="500">

### Fully Connected Layer

### Additional Stuff / Classification Layer /Full Pixel Layer?


## References

<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" width="22">
<center>{% include image.html url="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png"
description="Max Pooling. [CITEHERE3]" size="300" %}</center>
