---
layout: post
title: Installing Tensorflow
---
![Image of Yaktocat](https://www.tensorflow.org/images/logo-alt@2x.png)
In this post we'll go over installing Tensorflow using python. I prefer using the Jupyter notebook for my python scripts so we'll go through installing Anaconda(which includes Jupyter Notebook) and how to install Tensorflow using this environment.

### Step 1. Install Anaconda
https://www.continuum.io/downloads
If you don't have an NVIDIA gpu, skip to step 4

### Step 2. Download/install NVIDIA Toolkit
This can be downloaded [here](https://developer.nvidia.com/cuda-toolkit). After it is finished downloading run the executable to install it.

### Step 3. Download/install CUDNN
This can be downloaded [here](https://developer.nvidia.com/cudnn). After it has finished downloading extract the zip and move the bin, include, and lib folders to  _C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0_

### Step 4. Download/install Tensorflow
Currently Tensorflow only supports Python 3.5, though you can download an unofficial wheel for 3.6 [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

#### Python 3.5
- Open Anaconda Prompt
- type _conda install tensorflow-gpu_

#### Python 3.6
- Download wheel file
- Open Anaconda Prompt
- Type _pip install "directory of wheel file"_

### Verify Installation
There are two things we need to check to make sure tensorflow is correctly installed.

#### Tensorflow is installed
- Open Anaconda Prompt
- Type _python_
- Type _import tensorflow as tf_

If tensorflow is installed correctly then there shouldn't be any errors or issues.

#### GPU is recognized
- Follow above steps
- Type _sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))_

If a gpu is recognized its name should appear. If not then make sure you've performed steps 2 and 3 correctly.

### Additional Libraries
There are additional libraries that will need to be installed to run the tutorials
- conda install matplotlib
- conda install scipy
- conda install pandas


## Tensorflow Structure
Here we're going to go over a few concepts in tensorflow to get acclimated with how to use it. It took me a while to get comfortable with the library so I wanted to go over basics that took time for me to grasp. We'll make a simple FNN(Feed forward) structure and train it.

#### Things to discuss
- The graph
- Variables
- Placeholders
- What else?
