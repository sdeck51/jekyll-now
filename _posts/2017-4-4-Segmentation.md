---
layout: post
title: Tensorflow - Segmentation
---

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>



# Intro
This tutorial attempts to build a fully convolutional network along the lines of the paper "Fully Convolutional Networks for Semantic Segmentation"[1]. Some fair warning. This model needs roughly 6GB VRAM to run and can take days to train. My current model still has a ways to go towards replicating the FCN in the paper.

### What is Image Segmentation?
![](http://i.imgur.com/mSJDVCS.jpg)![](http://i.imgur.com/qZh484g.png)
In computer vision, image segmentation is the idea of partitioning an image into segments. These segments represent objects and boundaries that can be used to more easily label or classify what is in an image. Semantic segmentation is a variation on segmentation, where not only are we partitioning an image into coherent parts, but also labeling the parts.

### Fully Convolutional Networks
Fully Convolutional Networks(FCN) are fairly new architectures. Like ConvNets, FCNs are buit with convolution and pooling layers to extract feature maps from input images. What differs FCNs from traditional classification ConvNets is instead of classifying entire images FCNs classify every pixel. In a traditional ConvNet the fully connected layers at the end of the network removes spatial information about input to define an overal global classification. In a FCN instead of fully connected layers there are additional convolutional layers instead. This retains spatial information and allows the network to classify on a per pixel level. In this tutorial an FCN will be implemented using a model from the paper, along with the additional network components needed to perform semantic segmentation classification.

# Data
The data that is used comes from the PASCAL VOC2012 dataset. It contains nearly 1500 training and 1500 validation images that also have accompanying segmentation class labels. The dataset has 22 classes There is one problem in this dataset that affects the results between this and the original papers. Every object in the labels has an outline. This outline represents a void class. In calculating loss the model should ignore the pixels with this class. Tensorflow does not have the capability to ignore labels, and so my fix was to preprocess the void pixels into background pixels.

# Building the Model

If you've gone through the Facial Keypoint Detector tutorial then this will be fairly similar. Instead of building a custom network and iterating with different learning rates, regularization techniques and whatnot to optimize the model it will instead follow one set of parameters in the paper. My current code implements the model using the VGG19 model, though we will talk about the VGG16 model as they are quite similar.

<center>{% include image.html url="http://i.imgur.com/RYOXk1D.png"
description="VGG16 model above, FCN model below" size="800" %}</center>

VGG16 is a convolutional neural network built by Oxford Universty. It , along with a family of VGG models, was entered in  ILSVRC-2014 and won first and second place for localization and classification. It contains 16 layers of convolution, fully connected, and softmax layers. This model is going to be used as the basis for the fully convolutional network, as the paper uses. We'll load VGG16(19 for now) by defining each layer of the network in code, and then we'll use a file that contains preloaded data to set the weights of the network. Afterwards we'll use transfer learning and create a new network that is called the deconvolution network. 

## Deconvolution Network
The deconvolution network, or more correctly named upsampling network, takes input from a fully convolutional vgg network and performs multiple transpose convolutions. Transpose convolution can be seen as convolution with a fractional stride, or simply as upsampling the input into a larger featuremap. The goal of this layer is to construct the input as a semantically segmented image. In the paper by Long et all they use 3 transpose convolution layers, where two layers share similar shapes with earlier pooling layers in the VGG architecture, and the last layer outputs the final semantically segmented image. The first two layers have two pooling layers predictions added to it, as can be seen at the bottom of the figure. What is happening is the output of the vgg network is very small, in fact is 14x14 for a 224x224 input. This means the upsampling network is upsampling from 14x14 to 224x224. This is going to produce a very coarse result. What Long et al implemented to fix this problem was skip connections from the pooling layers. The pooling layers are put through 1x1 convolutions to predict the segmentation labels from those layers. Those predictions are then added to the prediction of the upsampling layer. This adds finer detail to the output. This process is implemented in the first two layers, and then the third layer outputs the final prediction. This prediction is of sie WxHx#classes. This 3d output is made up of one hot vectors for each pixel, where the 1 is the class number as an index. This can be put through an argmax function and the result is an image with each pixel having a class number.

# Training
Training this fully connected network has to be done somewhat carefully. One issue is that this network has over 144M parameters, and is nearly filling up all of the VRAM in my GPU. Because of this you can only run a few batches at a time. In the paper they run 20 batches, but for most of us we'll only be able to run between 1-3 at a time. This poses another issue. When obtaining training and validation loss we can once again only have 1-3 batches at a time. This increases the variance in the loss, which will result in a much more jagged curve. This can be combatted by simply obtaining the loss for multiple batches, however this will decrease the speed of training, and so you need to balance out the accuracy in your loss with the time it will take to finish. With a GTX 1070(I could not run this on my laptop) I had to run the machine for over 48 hours before beginning to get coherent segmentation predictions.


# Results
Results for this experiment show that I am not getting similar results to the paper. There are several reasons for this. My model does not correctly implement the scoring layers. Also my hardware is not capable of copying the original experiment. Their input uses 500x500 images with 20 images per batch while I can only use 224x224 images with 3 images per batch. Below are the results.


|  | Pixel Accuracy | Mean Accuracy | Mean IU| FW IU |
| (Long et al)[1] | 90.5 | 76.5 | 63.6 | 83.5 |
| Mine | 81.5 | 33.7 | 26.3 | 69.9 |
{: .tablelines}

<center>{% include image.html url="http://i.imgur.com/VqBKjR8.png"
description="Lamb Image" size="250" %}{% include image.html url="http://i.imgur.com/qg2xMWD.png"
description="Lamb Label" size="250" %}{% include image.html url="http://i.imgur.com/1jffKTs.png"
description="Lamb Prediction" size="250" %}</center>
<center></center>


References:
1. Long, J., Shelhamer, E., and Darrell, T. Fully convolutional networks for semantic segmentation.
CVPR 2015


## Full Code: 
https://github.com/sdeck51/CNNTutorials/blob/master/6.%20Segmentation_Tutorial/Segmentation2.ipynb
