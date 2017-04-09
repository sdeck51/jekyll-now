---
layout: post
title: Tensorflow - Facial Feature Detector
---
![](http://i.imgur.com/90KjE6A.png?2)



# Purpose
We want to build a model that can detect keypoints related to features on the face. Here I discuss how this is done. You can follow the concepts with your own preferred ML library, or you can download the code at the end and follow along.


# Data
Like all machine learning algorithms, data is crucial in building a regressor or classifier. This tutorial will use regression as we need to output a set of points that will define the positions of the facial keypoints. The data we'll be using can be found [here](https://www.kaggle.com/c/facial-keypoints-detection/data). It consists of 7049 96x96 grayscale images of faces. The labeling on the images are sparse though, so only a little over 2000 images have the full label. The labels for this set are 30 floating point values that represent 15 x and y coordinates to line up with facial features. The data is formatted in a single csv file.
<center>{% include image.html url="http://i.imgur.com/rPjZh9h.png"
description="data sample with labels overlaid" size="250" %}</center>

# The Model
We'll be looking at two different models. One will be a simple 2 layer neural network dubbed SimpleNet, while the other is a 6 layer convolutional neural net dubbed ConvNet. We'll be looking at the performance of the first to get baseline performance and then move over to ConvNet for further performance tweaks and adjustments.

<center>{% include image.html url="http://i.imgur.com/pVDFdO0.png"
description="SimpleNet" size="250" %}</center>

SimpleNet is a neural network with 2 layers, a 1000 neuron hidden layer, and a 30 neuron output layer. The input is of size 9126, which is 96x96, the size of the image. The output is the 30 (x,y) coordinates


<center>{% include image.html url="http://i.imgur.com/Y7VQEbL.png"
description="ConvNet" size="900" %}</center>

ConvNet is a 6 layer convolutional neural network. Obviously more complicated than SimpleNet, ConvNet has 3 convolution layers, each followed with pooling layers using max pooling. 3 fully connected layers follow the last pooling layer of size 1000, 1000 and 30. The model takes in a 96x96 image and outputs 30 (x,y) keypoint coordinates.

# Weight Initialization
To start off, initialization is key in creating a well fit model. The initial weights of the network can impact how the model will learn. The first experiment to look at is initializing the models weights to zero. Xavier initialization will also be looked at for comparison.

<center>{% include image.html url="http://i.imgur.com/oqrCUu9.jpg"
description="SimpleNet and ConvNet with zero and xavier initialization" size="900" %}</center>

From the graph above there is a clear difference between initializing the weights to be xavier and zero. In both cases the networks were trained using SGD with momentum and a learning rate of 0.01. The Xavier initialized models both continue to reduce in loss over time, while the zero initialized models halt around a loss of 0.0044. There are a few factors as to why this can be happening. As the model optimizes via the backwards pass, the weight of zero is passed to previous weights, which creates little to zero gradientsalso zero. If the gradient is zero then the neuron won't learn. Another reason is that the model is symmetric. This poses a problem, as once again when learning new features the neurons in similar layers will learn the same weights since the later layers all share the same weights among neurons. This means the model is learning redundant features. This combined with low gradients causes the model to stop learning.

Xavier initialization is a method of initializing the weights such that the variance across layers is the same. The reason this is needed is because as a signal propagates from the beginning layers, to the later layers, and back, we don't want that signal exploding or diminishing, and that relies on the gradients from the weights. Too small and the signal disappears. Too large and the signal may explode.  This means we want a similar distribution among the weights. In Understanding the Difficulty of Training Deep Feedforward Neural Networks[1] Glorot and Bengio discuss using uniform and gaussian distributions  with variance of  2 over the number of inputs and outputs.  In Delving Deep into Rectifiers He builds off Glorots work and suggests using 2 over the number of input when it comes to linear rectifiers which is what this network uses. Xavier isn't the only valid method of weight initialization. A gaussian distribution with zero mean and tuned variance is also used, though the advantage in Xavier initialization comes from the fact it scaling based on the input and or output, which allows us to worry less about tuning it.

# Optimizers
Since backpropagation involves minimization of a function using first order equations, we need an optimization method to calculate that. There are several different methods that can be used, and for this tutorial we'll look at three, namely stochastic gradient descent, stochastic gradient descent with momentum, and ADAM. 

### Stochastic Gradient Descent
Gradient Descent is perhaps the most simple algorithm for gradient finding. This is because it literally follows the gradient to optimize the function. In SGD you feed all of your data to the model and then the model takes a step based on all of the input. This is good for finding the best averaged direction, but is slow. In Stochastic Gradient Descent(A better name, more often used name is MiniBatch Gradient Descent) a batch of the input is applied and the optimizer updates based on that batch. This is good as the model can learn without needing to go through the entire data set, which makes it quicker to learn, and uses less memory as the entire dataset need not be ran.

<center>{% include image.html url="http://i.imgur.com/HkR8Pav.jpg"
description="SimpleNet loss using SGD" size="1000" %}</center>

### SGD with Momentum
Momentum(I'll refer to it as this for now on) is an addition to SGD. Along with following the gradient it also applies an update from the previous step, so if the optimizer is following a certain direction for multiple steps it will gain momentum and move further in that direciton.

<center>{% include image.html url="http://i.imgur.com/YJpNfFY.jpg"
description="SimpleNet loss using SGD with Momentum" size="1000" %}</center>

<center>{% include image.html url="http://i.imgur.com/S0QetKl.jpg"
description="ConvNet loss using SGD with Momentum" size="1000" %}</center>

### ADAM 
ADAM, or Adaptive Moment Estimation is an adaptive learning algorithm. With the previous methods you must set the learning rate that the optimizer will take steps at. In ADAM the expenentially decaying average of past square gradients and momentum are stored to estimate the moments and update the step size automatically. This is useful when you're not sure what you should be using for a learning rate. We'll see though that ADAM can have severe issues with overfitting.

<center>{% include image.html url="http://i.imgur.com/m6cWApN.jpg"
description="SimpleNet loss using ADAM" size="1000" %}</center>


<center>{% include image.html url="http://i.imgur.com/42qsCzG.jpg"
description="ConvNet loss using ADAM" size="1000" %}</center>
More information about the optimizers can be found on the concepts page, with equations. With these detailed lets look at how they handle optimizing SimpleNet at varying learning rates.


So with our weights initialized and a general idea about the learning rate, we can go ahead and run cross validation on a few models to verify the robustness across the dataset. However, there are additional techniques that can be used towards generating a better model. One problem with our model is the dataset is rather small. Of the 7000 images in the set only 2000 or so are being used, as only that many have full labels. This is somewhat of an issue, though there are ways to help alleviate the problem. While having more data is preferred, data augmentation is a method that helps generalize the training set by artificially increasing its size.

# Data Augmentation
Artificially augmenting data to increase the variation of it, such that it becomes a new image, but is still valid for the associating label, is a popular method in generalizing and regularizing a model. The original AlexNet paper[] demonstrated using augmentations such as translation and reflecting reduced model loss. There are several different methods for augmenting images. In this tutorial two methods will be employed. The first is reflecting images horizontally(along with the labels) to "double" the dataset. This makes the dataset "have" 4000 images now. Obviously not as good as an additional 2000 unique images, but better than without. The other method is image rotation, to help recognize images that have crooked heads. These smaller subsets in the data will become more recognized due to increasing the amount of images with those features. Below is a ConvNet with ADAM optimization employing rotation and reflection data augmentations.


<center>{% include image.html url="http://i.imgur.com/cbYRHyV.jpg"
description="ConvNet loss using ADAM with data augmentation" size="1000" %}</center>

This was an interesting graph that should be included here. ADAM is jumping in and out of regions in the function due to its learning rate. This type of issue can be solved by lowering the learning rate so it isn't consistently jumping across the function.

<center>{% include image.html url="http://i.imgur.com/X1eZw1Y.jpg"
description="ConvNet loss using ADAM with data augmentation, learning rate 1e-5" size="1000" %}</center>

With a new learning rate of 1e-5 the loss function appears much smoother. The mean loss is 0.000827 and the mean median is 0.000766. 


# Dropout
To further work towards generalization we can implement methods to regularize our system. Dropout is one such method in doing that. Popularized by AlexNet in 2012, dropout is a method that strips certain neurons from learning during training phases. The proposed idea behind how this works is that by stopping certain neurons from learning, other neurons will learn features the unlearnable ones are, and as this process runs more and more the neurons learn different features instead of learning the same features. 

<center>{% include image.html url="http://i.imgur.com/SbvWoA1.jpg"
description="ConvNet loss using ADAM with data augmentation, learning rate 1e-5" size="1000" %}</center>

The ADAM model appears to begun converging, though SGD with Momentum looks like it needs additional iterations. One interesting thing to note is that the distance between training and validation between Momentum is smaller than ADAM. This suggests that with additional epochs SGD with Momentum could reach a lower validation loss than ADAM.


<center>{% include image.html url="http://i.imgur.com/4cdH59S.png"
description="ConvNet loss using ADAM with data augmentation and dropout, learning rate 0.0001" size="1000" %}</center>

# Further Improvements
I end the tutorial here, however there are additional improvements that can be made to the model. Additional training time can reduce the loss somewhat. The reason I stopped at 2000 epochs is due to time. Running 500 epochs on my laptop is roughly 30 minutes, so the final ADAM and Momentum models both took over 10 hours each to run. With a better system, or simply more time to let it run it can be better. Another improvement that can be made is to add additional augmentations to the data. Things like jitter can help regularize the system. Lastly, generating separate models for each label would increase the accuracy of the system. From the data given, only 2000 of the 7000 images have full labels, so using separate models to be able to use all of the data will improve performance.

# Full Code
The full code I used can be found below. Folders may need to be made for storing models, as well as the data.
https://github.com/sdeck51/CNNTutorials/blob/master/7.%20FacialFeatureDetection_Tutorial/FaceDetector4.ipynb
