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

<center>{% include image.html url="http://i.imgur.com/Dibvwaa.png"
description="2d discrete convolution" size="550" %}</center>

The formal definition for 2d discrete convolution is above. If you want a more intuitive demonstration see the animation below.

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

# Activation Functions

## Sigmoid, Hyperbolic Tangent

## Linear Rectifiers

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

<center>{% include image.html url="http://i.imgur.com/QGnhYAi.png"
description="" size="200" %}</center>

You can see that this loss function increase the error exponentially the farther away it is from the label. Another popular function is MSE.

<center>{% include image.html url="http://i.imgur.com/DAa1S2c.png"
description="" size="200" %}</center>

So for a machine to predict a data's label correctly, it obviously needs a prediction close to the label. How do we go about doing this though? What needs to change in the machine to allow it to predict closer to a datas label? We need to minimize loss of the loss function. We can turn this into an optimization problem and try to find a minimum for this error. This leads us to the backward pass.

## Backward Pass
In the backward pass phase our goal is to take the loss function and minimize it with respect to the weights in our model. This means our model is dimensionality equal to the number of weights, which can be extremely large. This process calculates partial derivatives in each layer using the chain rule. It works its way backwards, layer through layer, calculating and updating new weight values using some optimization method with a first order method.

### Optimization Methods

<center>{% include image.html url="http://i.imgur.com/GM1LByj.jpg"
description="Batch Gradient Descent" size="200" %}</center>

There are several different gradient descent optimization methods. The most basic one is Batch Gradient Descent. In BGD optimization is performed using the entire training set. The advantage of using this is you will follow the exact gradient. This may not be wanted though as CNNs are not simply convex models, and so you may easly get stuck in a local minimum. Another disadvantage is this can be very slow when you have large datasets. In BGD theta represents the parameters, or weights we're wanting to change. J is the loss function. Eta is the learning rate parameter.  

<center>{% include image.html url="http://i.imgur.com/OmrgPQi.jpg"
description="Stochastic Gradient Descent" size="290" %}</center>

Another method is Stochastic Gradient Descent. The main difference between SGD and BGD is instead of requiring the entire dataset, you only use 1 sample from the set. Single sampling however has high variance with respect to the gradient direction so optimization can be very slow. Thus using an inbetween method is more optimal.

<center>{% include image.html url="http://i.imgur.com/Yr2d6Pq.jpg"
description="Mini-Batch Gradient Descent" size="360" %}</center>

Mini-Batch Gradient Descent is the most popular of the three methods to use. Unlike either, you can determine the sample size of training data to use instead of limiting it to either all or none. This provides a large performance boost over standard BGD, and having multiple samples averages out the gradient direction better than a single sample. This method is implemented in Tensorflow as tf.train.GradientDescentOptimizer.

For basic gradient descent optimization there are a few challenges that you will come across. One is the learning rate. A high learning rate results in a larger step size. This means you could converge to a minimum quicker by covering more ground per step. This can also do the opposite, where steps sizes step over statationary points, oscillate around them, or even diverge. On the other hand a small learning rate may causing training to take too long. One method to minimize these effects is to have a decaying learning rate, where it's scheduled to reduce in size. This can allow you to have a larger step size early on and then later a small step size is used which should help with convergence. Another problem is that these networks that are being built are high nonconvex. This means that there is more than one local minimum. For BGD and SGD it makes it very hard to escape when its approaching a zero gradient.

<center>{% include image.html url="http://i.imgur.com/KNIUuGJ.jpg"
description="Momentum" size="250" %}</center>

Another improvement that can be made is implementing momentum. Like rolling a ball down a hill, the previous direction that you move towards affects your next step. Gamma is the momentum value and is generally .9, and at most less than 1. This is implemented in Tensorflow as tf.train.MomentumOptimizer.

ADAM is a popular optimization technique that was published in 2014. [CITE] It stands for Adaptive moment Estimation and computes adaptive learning rates for each weight parameter. It works by keeping track of an exponetially decaying average of past first and second moments. 

<center>{% include image.html url="http://imgur.com/Du5zqCH.jpg"
description="ADAM [cite]" size="250" %}</center>

M and v represent the estimates of the first and second moments of the gradient while g is the gradient itself. Bias correct is done to them, resulted in the "hat" version. Then the weights are optimized using the last equation. This method is implemented in Tensorflow as tf.train.AdamOptimizer.

There are many more optimization algorithms that can be used for training your machine. The ones above are ones that are used in the other tutorials. From here you should have enough conceptual knowledge in how a CNN(or regular neural network) is trained.

# Validating/Testing

Once you're able to feed data into a machine, you want to know how well it's performing. As I mentioned earlier you want to split up your data such that you have training data, validation data and/or testing data. We want to see how good our machine is, against both the training and the validation/test data. The ratio between the two will tell you if your model is training well or if you're overfitting the data. If you're overfitting then you'll see a low training error with a high validation/test error. There are different splits you can use. If you have enough data splits tend to side more towards high training percentage. The Imagenet competition for example splits data into 1,200,000 training, 50,000 validation, and 100,000 testing. In the segmentation tutorial the dataset that is used is split 20,000 training and 2,000 validation. There isn't a concrete ratio to use. Just make sure not to cross contaminate the sets.

# Conclusion

This covers the basics in understanding convolutional neural networks, how they work, and how to use them. In the tutorials we'll cover how to program a models, how to load model, as well as how to load weights into a build model. The tutorials will cover 3 applications of CNNS, namely classification, segmentation and feature detection. Each tutorial has step by step instructions explaining what we are implementing and how to implement it using Tensorflow. My hope is that people will be able to read through this, gain a sliver of understanding of CNNs, be able to follow along in the tutorials, and then improve on them or make your own netoworks!



## References
I need to cite more things here.
ADAM
Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13
