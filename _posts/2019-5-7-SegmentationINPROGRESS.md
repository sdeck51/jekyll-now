---
layout: post
title: Tensorflow - Segmentation
---

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>

| P | Q | P * Q |
| - | - | - |
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | F |
{: .tablelines}
|====+====|
+====|====+
|=========|
|=

|---|---|---|
|a  | b | c|

| bff|sd|asd|
{: .tablelines}

|---|---|---|
|a  | b | c|
|---|---|---|
| bff|sd|asd|
{: .tablelines}

|---|---|---|
|a  | b | c|
|---|---|---|
| bff|sd|asd|
|---|---|---|
{: .tablelines}

|---+---+---|
+ :-: |:------| ---:|
| :-: :- -: -
:-: | :-
{: .tablelines}

|-----------------+------------+-----------------+----------------|
| Default aligned |Left aligned| Center aligned  | Right aligned  |
|-----------------|:-----------|:---------------:|---------------:|
| First body part |Second cell | Third cell      | fourth cell    |
| Second line     |foo         | **strong**      | baz            |
| Third line      |quux        | baz             | bar            |
|-----------------+------------+-----------------+----------------|
| Second body     |            |                 |                |
| 2 line          |            |                 |                |
|=================+============+=================+================|
| Footer row      |            |                 |                |
|-----------------+------------+-----------------+----------------|
{: .tablelines}
# Intro
This tutorial attempts to build a fully convolutional network along the lines of the paper "Fully Convolutional Networks for Semantic Segmentation"[cite].
![](http://i.imgur.com/ysw2ZZx.png?1)

# Purpose/Goal
The purpose of this tutorial is to demonstrate how to perform pixelwise classification using transpose convolution layers on a deep network. This process creates segmented images that can separate the class object in a scene.

### What is Image Segmentation?
![](http://i.imgur.com/mSJDVCS.jpg)![](http://i.imgur.com/qZh484g.png)
In computer vision, image segmentation is the idea of partitioning an image into segments. These segments represent objects and boundaries that can be used to more easily label or classify what is in an image. Semantic segmentation is a variation on segmentation, where not only are we partitioning an image into coherent parts, but also labeling the parts.

### Fully Convolutional Networks
Fully Convolutional Networks(FCN) are fairly new architectures. Like ConvNets, FCNs are buit with convolution and pooling layers to extract feature maps from input images. What differs FCNs from traditional classification ConvNets is instead of classifying entire images FCNs classify every pixel. In a traditional ConvNet the fully connected layers at the end of the network removes spatial information about input to define an overal global classification. In a FCN instead of fully connected layers there are additional convolutional layers instead. This retains spatial information and allows the network to classify on a per pixel level. In this tutorial an FCN will be implemented using a model from the paper, along with the additional network components needed to perform semantic segmentation classification.

# Data
The data that is used comes from the PASCAL VOC2012 dataset. It contains nearly 1500 training and 1500 validation images that also have accompanying segmentation class labels. The dataset has 22 classes There is one problem in this dataset that affects the results between this and the original papers. The data 
# Building the Model

If you've gone through the Facial Keypoint Detector tutorial then this will be fairly similar. Instead of building a custom network and iterating with different learning rates, regularization techniques and whatnot to optimize they model it will instead follow one set of parameters in the paper. The model will be built using the VGG19 network and it will be optimized using a learning rate of 0.0001 using SGD with Momentum. There are a few aspects that we cannot replicate however.

These are the basic ConvNet layers needed to build VGG19.

# VGG19

<center>{% include image.html url="http://i.imgur.com/KCylRbk.png"
description="data sample with labels overlaid" size="800" %}</center>

VGG19 is a convolutional neural network built by Oxford Universty. It , along with a family of VGG models, was entered in  ILSVRC-2014 and won first and second place for localization and classification. [1] It contains 19 layers of convolution, fully connected, and softmax layers. This model is going to be used as the basis for the fully convolutional network, as the paper uses. We'll load VGG19 by defining each layer of the network in code, and then we'll use a file that contains preloaded data to set the weights of the network. Afterwards we'll use transfer learning and create a new network that is called the deconvolution network. 

# Deconvolution Layer
The deconvolution layer, or more aptly named transpose convolution layer, takes input and performance transpose convolution. This can also be seen as convolution with a 1/k stride, where k is <=1. This essentially upsamples the input into a larger output. The goal of this layer is to construct the input as a semantically segmented image. In the paper by Long et all they use 3 transpose convolution layers, where two layers share similar shapes with earlier pooling layers in the VGG architecture, and the last layer outputs the final semantically segmented image. The first two layers get two pooling layers added to it. The reason for this is the input for the first deconvolution layer is very small, and upsampling that input will create a very coarse, blocky image. Earlier layers from VGG19 are fed to these two layers to help inject detail that was lost.


The network consists of 3 transpose convolution layers, with the first two recieving skip connects from the 3rd and 4th pooling layer in VGG19. The final transpose convolution layer is the output layer and the final layer in the entire network. This layer will output the predicted label as an image of class based pixels.

# Training
Training this fully connected network has to be done somewhat carefully. One issue is that this network has over 200M parameters, and is nearly filling up all of the VRAM in my GPU. Because of this you can only run a few batches at a time. In the paper they run 20 batches, but for most of us we'll only be able to run between 1-3 at a time. This poses another issue. When obtaining training and validation loss we can once again only have 1-3 batches at a time. This increases the variance in the loss, which will result in a much more jagged curve. This can be combatted by simply obtaining the loss for multiple batches, however this will decrease the speed of training, and so you need to balance out the accuracy in your loss with the time it will take to finish. With a GTX 1070 I had to run the machine for over 48 hours before beginning to get coherent segmentation predictions.


# Results
Results for this tutorial are the training and validation loss graphs as well as predictions. Below are some predictions from the machine I trained with their actual image and actual label. Each image comes from the validation set.
Actual Image

![](http://i.imgur.com/4SNhXib.png)

Actual Label

![](http://i.imgur.com/gdXqSpD.png)

Predicted Label

![](http://i.imgur.com/OYQFusp.png)

100k steps

Actual Image

![](http://imgur.com/Xs6wPpb.png)

Actual Label

![](http://imgur.com/2twn4A8.png) ![](http://imgur.com/WGWmOon.png)

Predicted Label

![](http://i.imgur.com/6aKWtEJ.png) ![](http://imgur.com/BtJHdlP.png)

another 100k steps

Actual Image

![](http://imgur.com/Cf9OvDD.png)

Actual Label

![](http://imgur.com/2psC6Dv.png)

Predicted Label

![](http://imgur.com/ZVykfPn.png)

As we can see it's starting to see bushes.

![](http://imgur.com/vFVNBXv.png)

![](http://imgur.com/Rgkqubs.png)

![](http://imgur.com/svHMXUF.png)

![](http://imgur.com/AyWvWf6.png)![](http://imgur.com/BhJCNm8.png)![](http://imgur.com/1XUpg8k.png)

References:
1. Long, J., Shelhamer, E., and Darrell, T. Fully convolutional networks for semantic segmentation.
CVPR 2015


## Full Code: https://github.com/sdeck51/CNNTutorials/blob/master/6.%20Segmentation_Tutorial/Segmentation2.ipynb
|-----------------+------------+-----------------+----------------|
| Default aligned |Left aligned| Center aligned  | Right aligned  |
|-----------------|:-----------|:---------------:|---------------:|
| First body part |Second cell | Third cell      | fourth cell    |
| Second line     |foo         | **strong**      | baz            |
| Third line      |quux        | baz             | bar            |
|-----------------+------------+-----------------+----------------|
| Second body     |            |                 |                |
| 2 line          |            |                 |                |
|=================+============+=================+================|
| Footer row      |            |                 |                |
|-----------------+------------+-----------------+----------------|

After
