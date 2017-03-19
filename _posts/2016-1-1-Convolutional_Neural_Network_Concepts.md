---
layout: post
title: Understanding Convolutional Neural Networks
---

<center>{% include image.html url="http://i.imgur.com/6Xe6Nz7.png" description="LeNET. [CITEHERE3]" size="900" %}</center>

# Introduction
Convolutional neural networks have recently gained traction as the go to method for several computer vision fields. In 2012 Alex Krizhevsky and his AlexNet won the yearly ImageNet Large Scale Visual Recognition Competition (ILSVRC), with a huge classification error record of 15% [CITE]. This was a rather big deal as the seconds place team had 26% error, which brought a center stage towards using convolutional neural networks. Ever since the 2012 competition convolutional neural networks has won the competition, and large companies like Google and Facebook have been using these networks for a host of their services. Their main use however, is in image processing and computer vision topics (though other problems sith spatial locality are also used!).

<center>{% include image.html url="http://imgur.com/9Npj4bi.jpg" description="Abraham Lincoln. [http://openframeworks.cc/ofBook/chapters/image_processing_computer_vision.html]" size="900" %}</center>


# Image Classification/Recognition
Image classification is a method of inputting an image, and outputting a class or probabilities of class that represents the image. Recognizing what things are is an inate and effortless ability of humans. When we look at a photograph we can easily discern what is in it, labeling each object automatically. Our ability to recognize easily patterns is not something that a computer can simply do. When a computer sees an image it has a 2 or 3 dimensional array of values representing pixel/color values. The goal in image classification is to take that data and have the computer determine the probability that it's some class. For example if I gave an image classifier a picture of a bird it could output .9 bird, .05 cat, .05 dog, which tells us that it's most probably a bird.

# What Convolutional Neural Networks do
Knowing that we're dealing with arrays of numbers and want our computer to classify those numbers, we need to think how to differentiate all of the unique features that makes a class a class. If I gave you an image of a dog how do you know it's a dog? You can tell by the features that make up a dog. This isn't something we really have to think about, but we know when we see 4 legs with paws, a long snout, and floppy ears, that we're probably looking at a dog. A computer needs to be able to determine these features as well. Convolutional neural networks are able to classify images by looking for features, such as simple curves and edges, and build these features up into abstract features through series of convolution layers.

# CNN Building Blocks
Typical convolutional neural networks are built out of only a few different layers. Generally when you feed an image into a CNN it goes through some combination of convolution, an activation function, pooling or downsample, and then flattened out through fully connected layer. For classification the output can be a class or probability of classes. In other applications, like feature detection, you can have positions that represent the location on the image of features. Before moving on to the tutorials we should have a good understanding of what these components are doing. Not only is it beneficial to understand for the sake of understanding, but it also helps when having trouble working with a model and knowing what each component is doing for debugging purposes.

# Convolution Layer
CNNs always start with a convolution layer. Our input image is some (w,h,c) size, where w=width, h=height, and c = channel length. As is in the name, this layer performs convolution on the input image. What is convolution though? In terms of images, which are 2 dimensional(well 3 as there's also the 3 color channels, but let's ignore that for now), imagine we have an 5x5 array. Each position in the array represents a pixel value. Now image we have a 3x3 array. We are going to take this array and slide it across the image. At each position this 3x3 array is at, we want to take the product of overlapping values and add them up. This array is a called a filter, or sometimes a kernel. 

<center>{% include image.html url="http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif"
description="2 dimensional convolution. [Feature extraction using convolution, Stanford]" size="500" %}</center>

The above animation demonstrates convolution. In green we have the input array. We can see a yellow array sliding across the input array. This area is known as the receptive field. For each shift it calculates a sum of products. In the animation the red text represent values of the kernel. These values are known as weights or parameters. We can see that after convolving the input with the filter we end up with a smaller array. This makes intuitive sense as there are only 9 unique locations the kernel can overlap the input array. This final output has different names. I like the term feature map. This is the basic theory and terminology used in the convolution layer. Next we'll go more in depth with convolution.

## Convolution Stride and Padding
When convolving an input there are two main parameters that can be adjusted. These are the stride and the padding. These are both useful parameters to adjust in reducing output size or retaining more information from the input.

### Stride
The stride dictates the distance the kernel moves across the input. In the animation from the convolution layer section we saw that the kernel shifted 1 pixel when sliding. This amount is the stride, which is 1. We can choose to increase the stride which will grant us a smaller output. This will also reduce spatial information.

<center>{% include image.html url="http://imgur.com/WdOj0NP.jpg"
description="Convolution with stride 1." size="450" %}</center>

In the above image we have 7x7 input array. Imagine we have a kernel of size 3x3. We can see that with a stride of 1 that we'll end up with a 5x5 output array. We can also see the receptive fields overlap with 2/3rd shared information.

<center>{% include image.html url="http://imgur.com/C7K9Y1O.jpg"
description="Convolution with stride 2." size="450" %}</center>

Now let's say that we have the same input and kernel, but with a stride of 2. This results in less overlap between receptive fields, whcih means less shared spatial information. On top of this, this will reduce the output array to 3x3. 

### Padding
The convolution figures we've seen so far contain padding of zero. Padding simply extends the image so the receptive fields can overlap outside of the image. Why is this important? If we were to continuously convolve an input image it would get smaller and smaller. What if we want our output to remain the same size as our input? Say we have a 5x5 input array. We can pad the outside with 1 layer of pixels.

<center>{% include image.html url="http://imgur.com/s5hJM62.jpg"
description="zero padding and replicated padding." size="450" %}</center>

Above shows such an example. On the left if we were to apply a 3x3 kernel with stride equal to 1, we end up with a 5x5 output, unlike a 3x3 output if we didn't padding. In the left case we pad the exterior of the array with zeros, so when convolving we simply have zero for those products. This isn't the only way of padding though. On the right side is an example of replicated padding. Instead of having zero on the edge you copy the neighboring edge pixel into the padding. These are the more popular options that are used in padding. Padding isn't also just used for increasing the size. If you have a filter with stride greater than one that doesn't evenly divide the array size then you'll need to pad the input.

<center>{% include image.html url="http://imgur.com/poCnN6D.jpg"
description="zero padding to extend side." size="300" %}</center>

# Feature Maps
With the understanding of how convolution and convolution layers work we can start to understand what exactly a feature map is.
This needs a good example to demonstrate feature activation

# Pooling Layer
Pooling, also known as subsampling, reduces the dimensionality of the previous layers feature maps. These layers also help with overfitting by filtering out higher frequency information and retaining general low frequency information that is shared across a distribution of data, rather than specific training data. Like convolution, pooling uses a sliding window technique. The difference is in the operation of the window. There are several different types of downsampling used in pooling layers, some more popular such as max pooling.

<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" height="400" width="500">


# Fully Connected Layer
After convolving to find features running through activation functions, and subsampling the feature maps repeatedly full connected layers are appended to the end of convolutional neural networks. Where convolutional layers connect input to output based on receptive fields in the input, fully connected layers connect every input in a given input volume to every output. You can think of them as matrix multiplying the input with the layers given weights to produce the output. Unlike convolution layers that have volume in their input and output, (remember, an image has width and height, but also is generally in color, adding a third dimension) fully connected layers flatten out input into a vector. These layers are used for finalizing a model before predicting. For example, if we had a dataset that had 10 classes, we would want to end with a fully connected layer that has 10 outputs, representing each class.

# Training a Network
So far we've discussed what makes up a typical convolutional neural network, as well as what the components are doing. This is barely scratching the surface, as for these networks to work we need to have features. How does a CNN obtain these features? It needs to learn them. This is done through training the network via a process called backpropagation.

When a CNN is newly built it has no information about anything. If you were to put an image into it it would spit out garbage. We need to teach the machine about the data we want it to classify. You can think about it as teaching anyone about anything new. You show your friend some new tech item. He/she knows nothing about it, but you tell him/her what it is. Like this process we need to have a process of telling our machine what the data we are feeding it is. This means that we ourselves need to know what the data is. Datasets used for training these types of networks have labels that are used to identify the class something belongs to. The idea of introducing information, having someone/something known/not know what it is, and then correcting it by telling them what it is essentially what backpropagation does. 

## Learning Data
Before diving into the backpropagation process we need to understand a few things. When training or teaching a CNN(or any supervised learning system) we need to have separate batches of data. We can easily train a network with a single image and have it correctly classify it. What we want to do though is have a machine that can correctly classify data it has never seen before. This means instead of learning the images specifically it is fed, it learns generalized features that can be used for new data. Networks that don't have enough data to train with generally overfit, which is when it does not learn generalized features, and only works well with the data it was fed with. Generally data is split up into training and validation, or training, validation, and testing. A lot of machine learning algorithms benefit from cross validation, which is where data is split up, and several models are made, each with different training/validation/testing data. This helps with overfitting as you get to sample different portions of data to train your machine. For convolutional neural networks this isn't as advisable to do. Data isn't as scarce, so overfitting isn't as large of an issue. Also with large datasets and large machine training time becomes an issue. It doesn't make sense to create multiple different machines trained on different portions of data that each take months to run. The key point here though, is never contaminate data you're training with data you're using to validate or test.

## Forward Pass
In the forward pass we'll take an image from our dataset and push it through the entire network. Since the network hasn't learned anything yet, we can say that it outputs an even distribution of values, say [.105, .0995, .11, .09, .1, .1, .099, .101, .1, .1]. This output is called a prediction, meaning it thinks that the image is .105 class1, .0995 class2 etc. This is obviously a bad answer but that's because the network doesn't know about any of the features that makes up this data, so it cannot classify well currently. This is fine though, as we're going to take this prediction and pass it to a loss function.

## Loss Function
The loss function is a function used to determine the error between the predicted class and the actual class. Let's say our machine is classifying handwritten digits, and each class is 0 - 9. The expected output for 6 would be [0,0,0,0,0,0,1,0,0,0]. If we had a somewhat trained machine and had [.05,0,0,0,0,0,.8,0,.05,.1] as the prediction we would want to compare the error. The loss function does this and there are several different types that can be used. A popular method for classification is cross entropy.

*cross entropy equation*

You can see that this loss function increase the error exponentially the farther away it is from the label. Another popular function is MSE.

*MSE equation*

So for a machine to predict a data's label correctly, it obviously needs a prediction close to the label. How do we go about doing this though? What needs to change in the machine to allow it to predict closer to a datas label? We need to minimize loss of the loss function. We can turn this into an optimization problem and try to find a minimum for this error. This leads us to the backward pass.


## Backward Pass
In the backward pass phase our goal is to take the loss function and minimize it with respect to the weights in our model. This means our model is dimensionality equal to the number of weights, which can be extremely large. This process calculates partial derivatives in each layer using the chain rule. It works its way backwards, layer through layer, calculating and updating new weight values using various  


## Optimization Algorithms

### Stochastic Gradient Descent




# IGNORE EVERYTHING BELOW
# What are Convolutional Neural Networks?
There are a lot of concepts in Convolutional Neural Networks(ConvNet) that need to be covered. here we'll discuss concepts and theory needed for understanding the Tensorflow tutorials. To start off ConvNets can be thought of as two concepts, one being a neural network and the other being organized via convolutions. We'll discuss what neural networks are and how they are trained. Then we'll look at what makes convolutional neural networks different from standard neural networks.

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
<center>{% include image.html url="http://i.imgur.com/6Xe6Nz7.png" description="LeNET. [CITEHERE3]" size="900" %}</center>
Moving on from perceptrons and simple neural networks we get into convolutional neural networks (ConvNet). These networks, as their name describes, use convolution through how the weights are tied between layers. Each pixel in each channel of a ConvNet represents a single input, so you can imagine these networks are fairly large. Their convolution layers allow for less interconnections between layers as weights are shared rather than are unique, and the structure retains spatial information since it's convolution. Let's quickly go over what convolution is, and look at the main components used to build a ConvNet.

### Convolution

<center>{% include image.html url="http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif"
description="2 dimensional convolution. [Feature extraction using convolution, Stanford]" size="500" %}</center>
A conv B =  sumx[n]h[n-k]

#### Kernel Sizes, Strides
In convolution you can define both the kernel size as well as stride. The kernel size is the height/width of the kernel and the stride is how far it slides while performing convolution. Along with this there are methods you can employ to the data you are convolving with. 

<center>{% include image.html url="http://imgur.com/WdOj0NP.jpg"
description="Convolution with stride 1." size="300" %}</center>



<center>{% include image.html url="http://imgur.com/C7K9Y1O.jpg"
description="Convolution with stride 2." size="300" %}</center>

At the edge of an image you need to decide if you want your kernel to lap over the edge of the image or to remain in side. If the former then a padding technique needs to be employed, which will extend the input data such that the kernel has additional data to convolve with.



### Convolution Layer
*feature map picture?*
Convolution layers consist of learnable kernels that are convolved against the input and output feature maps.  What this means is a layer has multiple kernels, or groups of weights, that are applied to the image.

#### Pooling Layer
Pooling, also known as subsampling, reduces the dimensionality of the previous layers feature maps. These layers also help with overfitting by filtering out higher frequency information and retaining general low frequency information that is shared across a distribution of data, rather than specific training data. Like convolution, pooling uses a sliding window technique. The difference is in the operation of the window. There are several different types of downsampling used in pooling layers, some more popular such as max pooling.

Show image of max pooling.
<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" height="400" width="500">

### Fully Connected Layer
Fully connected layers are simply layers where every input is connected to every node, hence fully connected. In convolutional neural networks these layers are placed towards the end to flatten out the model and eventually reach an output vector of 1 by X, where X is the number of classes.

### Additional Stuff / Classification Layer /Full Pixel Layer?


## References

<img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png" width="22">
<center>{% include image.html url="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-3-38-39-am.png"
description="Max Pooling. [CITEHERE3]" size="300" %}</center>
