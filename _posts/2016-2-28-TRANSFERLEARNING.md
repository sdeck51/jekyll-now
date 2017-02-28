---
layout: post
title: Transfer Learning using Inception v3.
---

In this post I'll go over how you can harness a deep neural network and classify your own images quickly!

# Intro
Here we learn how we can transfer learned features from one dataset to another.
#Purpose/Goal
The purpose of this tutorial is to introduce the concept of transfer learning. Transfer learning is a powerful tool in taking advantage of very deep networks trained on millions of images and using it for ones own classification. The problem with using a deep network on your own is that fully training the system can takes weeks. Using a network that is pretrained is one solution, however that network is made to classify the data it trained on. What if you have a set of classes you want to classify, and you need a large network to do it? You could train from scratch, but you could also use transfer learning and take advantage of a pretrained network. Using transfer learning will allow you to quickly train your data.
# Data
For the data in this tutorial we'll be looking at JPEG files. One problem I have training models is sometimes half of the work is just getting the data into your program. For this we'll implement a simple interface where data will be accessed in folder that will define the class of the image. So for example if you have a lot of images of planes you'll place them in a folder marked planes. Then the code will use the folder name as the class name. I'm going to demonstrate transfer learning using a few different datasets. In one we have 600 images of 6 different types of birds. Another we'll see what kind of accuracy we can get classifying male and female faces using a cropped celebrity database. The last one has 157 classes of random objects. 

# Process
### Things to discuss
- Loading a model (using Inception v3)
- loading weights
- bottleneck layer
- caching image bottlenecks
- implementing new classification layer

### Progress
Code works! Just need to format results.

# Results
I have three different data sets. I need to upload % accuracy. Would be good to pick out a random batch and show their classification.