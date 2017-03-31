---
layout: post
title: Tensorflow - Facial Feature Detector
---
![](http://i.imgur.com/90KjE6A.png?2)

In this post I go over how to make a facial feature detector. Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/7.%20FacialFeatureDetection_Tutorial/FaceDetector4.ipynb).

# Purpose
The purpose of this tutorial is to discuss several topics needed in understanding how to model a convolutional neural network. This ranges from simple topics such as weight initialization to regularization techniques for generalizing data. We'll also discuss ways to further improve the model. The code used can be found [here](https://github.com/sdeck51/CNNTutorials/blob/master/7.%20FacialFeatureDetection_Tutorial/FaceDetector4.ipynb).

# Data
Like all machine learning algorithms, data is crucial in building a regressor or classifier. This tutorial will use regression as we need to output a set of points that will define the positions of the facial keypoints. The data we'll be using can be found [here](https://www.kaggle.com/c/facial-keypoints-detection/data). It consists of 7049 96x96 grayscale images of faces. The labeling on the images are sparse though, so only a little over 2000 images have the full label. The labels for this set are 30 floating point values that represent 15 x and y coordinates to line up with facial features. The data is formatted in a single csv file.
<center>{% include image.html url="http://i.imgur.com/rPjZh9h.png"
description="data sample with labels overlaid" size="250" %}</center>

# The Model
We'll be looking at two different models. One will be a simple 2 layer neural network dubbed SimpleNet, while the other is a 6 layer convolutional neural net dubbed ConvNet. We'll be looking at the performance of the first to get baseline performance and then move over to ConvNet for further performance tweaks and adjustments.

# Weight Initialization
In neural networks, weights are the parameters that get adjusted during optimization. The initialization of these weights can make or break a model. Let's look at what happens when SimpleNet is initialized with weights equal to zero.

<center>{% include image.html url="http://i.imgur.com/h9Fp9w5.jpg"
description="SimpleNet with zero initialized weights" size="600" %}</center>

Since we don't have anything to compare it with it doesn't tell us much. Here's an equivalent SimpleNet model, but the weights are set using Xavier initialization.

<center>{% include image.html url="http://i.imgur.com/WM4Ia55.jpg"
description="SimpleNet with Xavier initialized weights" size="600" %}</center>

As you can see, the Xavier initialized model is at a much lower training and validation loss. So why is the zero initialized model so much worse off? The reason is because at a weight of zero the gradient is also zero. The machine learns through updating the model using a delta of the gradient. If that gradient is zero then it doesn't learn. Too large of a gradient and the weight may explode.  This means we want a similar distribution among the weights. In Understanding the Difficulty of Training Deep Feedforward Neural Networks[1] Glorot and Bengio discuss using a uniform distribution  from -1/sqrtn to 1/sqrt(n), where n is the size of the previous layer. This is called Xavier initialization which is what is being used in the second graph. There are other ways to initialize weights, such as using a gaussian distribution with zero mean and some variance.

# Optimizers
Since backpropagation involves minimization of a function using first order equations, we need an optimization method to calculate that. There are several different methods that can be used, and for this tutorial we'll look at three, namely stochastic gradient descent, stochastic gradient descent with momentum, and ADAM. 

### Stochastic Gradient Descent
Gradient Descent is perhaps the most simple algorithm for gradient finding. This is because it literally follows the gradient to optimize the function. In SGD you feed all of your data to the model and then the model takes a step based on all of the input. This is good for finding the best averaged direction, but is slow. In Stochastic Gradient Descent(A better name, more often used name is MiniBatch Gradient Descent) a batch of the input is applied and the optimizer updates based on that batch. This is good as the model can learn without needing to go through the entire data set, which makes it quicker to learn, and uses less memory as the entire dataset need not be ran.

<center>{% include image.html url="http://i.imgur.com/14P3pNk.jpg"
description="SimpleNet loss using SGD" size="600" %}</center>


### SGD with Momentum
Momentum(I'll refer to it as this for now on) is an addition to SGD. Along with following the gradient it also applies an update from the previous step, so if the optimizer is following a certain direction for multiple steps it will gain momentum and move further in that direciton.

<center>{% include image.html url="http://i.imgur.com/YJpNfFY.jpg"
description="SimpleNet loss using SGD with Momentum" size="600" %}</center>

### ADAM 
ADAM, or Adaptive Moment Estimation is an adaptive learning algorithm. With the previous methods you must set the learning rate that the optimizer will take steps at. In ADAM the expenentially decaying average of past square gradients and momentum are stored to estimate the moments and update the step size automatically. This is useful when you're not sure what you should be using for a learning rate. We'll see though that ADAM can have severe issues with overfitting.

<center>{% include image.html url="http://i.imgur.com/m6cWApN.jpg"
description="SimpleNet loss using ADAM" size="600" %}</center>

More information about the optimizers can be found on the concepts page, with equations. With these detailed lets look at how they handle optimizing SimpleNet at varying learning rates.



Let's also look at how they handle ConvNet.




Say some things about both, move on to only using ConvNet.

So with our weights initialized and a learning rate picked out, we should be ready to let a model train via cross validation, and verify the robustness of the models right? Well, there are a few more things we can do to improve our model beyond simply training it longer. One problem with our model is the dataset is rather small. I mentioned earlier that the dataset has over 7000 images. We're only using 2000 or so as only that many have full labels. This is somewhat of an issue so we should come up with a way to combat it. More data is generally always better. One method we can implement is data augmentation.

# Data Augmentation
When your dataset is too small one way to artificially increase it is to make variations of the data, such that it becomes a new image, but is still valid for the associating label. In our case there are many forms of data augmentation we can do. In this tutorial I'll perform two different augmentations. The first is flipping the images horizontally(along with the labels) to double the dataset. This makes our dataset "have" 4000 images now. Obviously not as good as an additional 2000 unique images, but better than without. The other method is image rotation. The reason I've added this is because in the dataset there are faces that are tilted, or crooked. These images make up a small subset of the overall set and so they are harder to get right, so implementing face rotations will teach the machine many more faces that have angled eyes, and mouths.

Lets take our best learning rate example from ConvNet and apply data augmentation to it.

Look at that! The training loss is worse, but validation loss is much better. This means the data we have is more generalized towards the unseen validation data. We want to work towards making the training and validation loss as close as can be, as that suggests our training data represents a distribution similar to the validation data.

# Dropout
To further work towards generalization we can implement methods to regularize our system. Dropout is one such method in doing that. Popularized by AlexNet in 2012, dropout is a method that strips certain neurons from learning during training phases. The proposed idea behind how this works is that by stopping certain neurons from learning, other neurons will learn features the unlearnable ones are, and as this process runs more and more the neurons learn different features instead of learning the same features. 

# Momentum Increase
