---
layout: post
title: Segmentation
---

In this post I'll show how you can perform segmentation using CNNs!

# Intro
For this tutorial I'm following the paper [here](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). Fair warning, if you want to run this you'll need a gpu with at least 8GB, otherwise you'll have to use the cpu version which will take eons to finish(I'm running a 12 hour training test atm and will probably have a terrible model).

![](http://i.imgur.com/ysw2ZZx.png?1)

# Purpose/Goal
The purpose of this tutorial is to demonstrate how to do per pixel classification using transpose convolution layers on a deep network. This process creates segmented images that can separate the classe object in a scene.
#Data
The data I'm using is from the MIT Scene Parsing website [here](http://sceneparsing.csail.mit.edu/). It contains 20,000 training images, and 2000 validation images. The different shades in the labeled image represents different classes. (Put in a color version)

![](http://i.imgur.com/mSJDVCS.jpg)![](http://i.imgur.com/qZh484g.png)

# Process

### Things to discuss
- Loading the data
- Building the Model (VGG 16/19)
- Loading the weights
- Implementing new classifier via transfer learning
- Transpose Convolution Layers - new layer type
- Fuse Layers/skip connections - adding previous layers to future layers

#### Progress
I have I believe working code. Once I get some results I'll be able to determine if I've made errors or not.

# Results
Nothing yet

# Random Stuff
Implemented Code seems to be working. I've built the VGG19 CNN with preloaded weights, and then added the additional "deconvolution"(Note this is wrong) network to structure the output back into an image. Issue right now is letting it train for a long enough time. This is the largest network I've built in tensorflow, and I'm actually running out of gpu memory with regular batch sizes, so as a result I'm having to reduce batch sizes which is increasing time to train. Currently I just need to let it train for a long time. 

I'm using the MIT Parsing Dataset. In the paper they state they received best results with at least 175 epochs. The MIT dataset has ~20,000 labeled images, while PASCAL VOC only has 3,000 with segmentation labels. So my plan is to run roughly the same amount at 175 epochs as the paper suggests, and then see what the results are.

I tried a quick run through to see if I would get at least blobs or something, though I think I need to run it longer than the 10,000 steps I did. Below are results of that. It looks like its forming the mountain, though with the wrong class. I'm currently running a "short" training run which is going to take 12 or so hours. I'm hoping to get blobs that match up, and not necessarily accurate silhouette. 

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

My impressions here are that I still have a ways to dgo however the splotch of red in the top left corner and splotch of blue on the right side indicates its starting to learn. From the steps I've taken on it so far it has gone down a magnitude in training loss so I need to just let it run for an extended amount of time. Model saving/loading is working and will be pushed into the next build, so I can let it run overnight this week. I'll also try to get the models guess on training samples as well as the validation samples above.

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

![](http://i.imgur.com/AyWvWf6.png)

![](http://imgur.com/OxmzT4V.png)

![](http://imgur.com/RgiBdkR.png)
