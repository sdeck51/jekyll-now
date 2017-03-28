---
layout: post
title: Tensorflow - Segmentation
---

In this post I'll show how you can perform segmentation using CNNs! Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/6.%20Segmentation_Tutorial/Segmentation2.ipynb) Under construction!

# Intro
For this tutorial I'm following the paper [here](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). Fair warning, if you want to run this you'll need a gpu with at least 8GB, otherwise you'll have to use the cpu version which will take eons to finish(I'm running a 12 hour training test atm and will probably have a terrible model).

![](http://i.imgur.com/ysw2ZZx.png?1)

# Purpose/Goal
The purpose of this tutorial is to demonstrate how to perform pixelwise classification using transpose convolution layers on a deep network. This process creates segmented images that can separate the class object in a scene.

### What is Image Segmentation?
![](http://i.imgur.com/mSJDVCS.jpg)![](http://i.imgur.com/qZh484g.png)
In computer vision, image segmentation is the idea of partitioning an image into segments. These segments represent objects and boundaries that can be used to more easily label or classify what is in an image. Semantic segmentation is a variation on segmentation, where not only are we partitioning an image into coherent parts, but also labeling the parts.

### Fully Convolutional Networks
Fully Convolutional Networks(FCN) are fairly new architectures. Like Convnets, FCNs impose convolution and pooling layers to extract feature maps from input images. What differs FCNs from traditional classification ConvNets is instead of classifying entire images FCNs classify every pixel. There are no fully connected layers, which are instead replaced with convolution layers. For this tutorial we're also implementing transposed convolution layers, used for upsampling. Lik convolution layers weights, upsampling layers weights are also learned. Additionally, skips connections are used in various layers towards the upsampling layer to hopefully capture finer grain features in the image. In this tutorial we'll attempt to build a FCN for semantic segmentation using the popular VGG19 model.

# Data
The data we'll be using is from the MIT Scene Parsing website [here](http://sceneparsing.csail.mit.edu/). It contains 20,000 training images, and 2000 validation images across 151 different classes. The data we need is simply formatted in 4 folders that contain training images, training labels, validation images, validation labels. Due to having so many images it's a good idea to cache them on disc for quicker access.

The first thing we need to do is collect the names of the files we're reading in.

{% highlight python %}
def list_files(dire):
    r= []
    names = []
    for root, dirs, files in os.walk(dire):
        for name in files:
            r.append(dire+'/' + name)
    return r

training_list= list_files(os.getcwd() + '/data/MIT_SceneParsing/ADEChallengeData2016/images/training')
validation_list= list_files(os.getcwd() + '/data/MIT_SceneParsing/ADEChallengeData2016/images/validation')

training_label_list= list_files(os.getcwd() + '/data/MIT_SceneParsing/ADEChallengeData2016/annotations/training')
validation_label_list= list_files(os.getcwd() + '/data/MIT_SceneParsing/ADEChallengeData2016/annotations/validation')
{% endhighlight %}

This will allows us to then grab each image in the dataset for training and testing purposes.

{% highlight python %}
def imageToArray(filename, size):
    image = misc.imread(filename) #include sklearn.misc to use!
    resize_image = misc.imresize(image, [size, size], interp='nearest')
    
    return np.array(resize_image)

def loadDataSet(data_list, label_list):
    dataset = []
    dataset_labels = []
    for filename in data_list:
        dataset.append(np.array(imageToArray(filename, image_size)))
    for filename in label_list:
        dataset_labels.append(np.array(imageToArray(filename, image_size)))

    return dataset, dataset_labels
    
{% endhighlight %}

Once you have this code up we can pull the data and labels for each set. This can take several minutes for the training images and labels.

{% highlight python %}
training_data, training_labels = loadDataSet(training_list, training_label_list)
validation_data, validation_labels = loadDataSet(validation_list, validation_label_list)
{% endhighlight %}

After 5-10 minutes or so the data should be fully loaded. Before moving on I found it to be a good idea to cache the files in a pickle file. I've had Jupyter Notebook crash which caused me to have to repull the data and waiting each time is aggravating.

{% highlight python %}
import pickle

def cacheData(data_name, label_name, data, label):
    with open(data_name, 'wb') as f:
        pickle.dump(data, f)
    with open(label_name, 'wb') as f:
        pickle.dump(label, f)
        
def loadData(data_name, label_name):
    data = pickle.load(open(data_name, 'rb'))
    label = pickle.load(open(label_name, 'rb'))
    return data, label
    
{% endhighlight %}

So what we'll do is take the data and label of each set and dump them to disk. This will significantly reduce the laod time on the training set.

{% highlight python %}
cacheData('valid_data.pkl', 'valid_labels.pkl', validation_data, validation_labels)
cacheData('training_data.pkl', 'training_labels.pkl', training_data, training_labels)
{% endhighlight %}

Now that we have saved the data to pickle files, whenever we need to reload the data we can use laodData instead of loadDataSet.

One additional change to the data that I made was that there are 4 images in the training set that are actually grayscale. This was causing a bug in my model, as my model only expects rgb channel images, so one last change to the data that I make is simply converting the 4 images to 3 channels and copying the originaly single channel to each. Also some reshaping is done to the labels.

{% highlight python %}
lt = len(training_data)

for i in range(0, lt):
    b = np.array(training_data[i])
    if b.shape == (image_size, image_size):
        print(str(i) + " " + str(b.shape))
        temp = np.empty([image_size, image_size, 3])
        temp.shape
        temp[:,:,0] = training_data[i]
        temp[:,:,1] = training_data[i]
        temp[:,:,2] = training_data[i]
        training_data[i] = temp
        
training_labels = np.reshape(training_labels, [-1, image_size, image_size, 1])
validation_labels = np.reshape(validation_labels, [-1, image_size, image_size, 1])
{% endhighlight %}

You should now be able to access all of the data.
# Building the Model

If you've gone through my Facial Feature Detector tutorial then this will be fairly similar. The main difference is that instead of building a custom model we're going to mimic the VGG 19 model. The reason we're doing this is because the paper mentioned at the beginning lists a few different networks that were used, including VGG 19. We also have access online to pretrained weights so we won't have to train the network from scratch.

For building the model I use a few generator functions to generate each layer. This helps reduce code and makes things overall cleaner and easier to read.

{% highlight python %}
def createConvolutionLayer(x_input, _weights, _biases, kernel_size, features, depth):
    # createConvolutionLayer generates a convolution layer in the session graph
    # by assigning weights, biases, convolution and relu function
    #
    # x_input - output from the previous layer
    # kernel_size - size of the feature kernels
    # depth - number of feature kernels
    #
    # returns convolution layer in graph
    #
    init = tf.constant_initializer(np.transpose(_weights, (1, 0, 2, 3)), dtype=tf.float32)
    weights = tf.get_variable('weights', initializer=init,  shape=_weights.shape)
    
    init = tf.constant_initializer(_biases.reshape(-1), dtype=tf.float32)
    biases = tf.get_variable('biases', initializer=init,  shape=_biases.reshape(-1).shape)
    
    #print("conv: input size: " + str(x_input.get_shape()))
    convolution = tf.nn.conv2d(x_input, weights, strides=[1,1,1,1], padding='SAME')
    
    print("w: " + str(weights.get_shape()))
    print("b: " + str(biases.get_shape()))
    added = tf.nn.bias_add(convolution, biases)
    
    return tf.nn.relu(added)
{% endhighlight %}
{% highlight python %}
def createFullyConnectedLayer(x_input, width):
    # createFullyConnectedLayer generates a fully connected layer in the session graph
    # 
    # x_input - output from previous layer
    # width - width of the layer (eg for a 10 class output you need to end with a 10 width layer
    #
    # returns fully connected layer in graph
    #
    #print("fc: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[x_input.get_shape()[1], width],
                             initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[width], initializer=tf.constant_initializer(0))
     
    matrix_multiply = tf.matmul(x_input, weights)
    
    return tf.nn.bias_add(matrix_multiply, biases)
{% endhighlight %}

{% highlight python %}
def createLinearRectifier(x_input):
    # createLinearRectifier generates a ReLu in the session graph
    # 
    # The reason this exists is due to the last fully connected layer not needing a relu while others do
    # x_input - output from previous layer
    # width - width of the layer
    #
    # returns ReLu in graph
    # 
    
    return tf.nn.relu(x_input)
{% endhighlight %}

{% highlight python %}
def createPoolingLayer(x_input, kernel_size, pool_type):
    # createPoolingLayer generates a pooling layer in the session graph
    # 
    # The reason this exists is due to the last fully connected layer not needing a relu while others do
    # x_input - output from previous layer
    # kernel_size - size of the kernel
    #
    # returns pooling layer in graph
    # 
    #print("pool: input size: " + str(x_input.get_shape()))
    #change to average pooling
    if pool_type == 'max':
        return tf.nn.max_pool(x_input, ksize=[1, kernel_size, kernel_size, 1], strides=[1,kernel_size, kernel_size, 1], padding='SAME')
    else:
        return tf.nn.avg_pool(x_input, ksize=[1, kernel_size, kernel_size, 1], strides=[1,kernel_size, kernel_size, 1], padding='SAME')
{% endhighlight %}

These are the basic ConvNet layers needed to build VGG19.

# VGG19

<center>{% include image.html url="http://i.imgur.com/KCylRbk.png"
description="data sample with labels overlaid" size="800" %}</center>

VGG19 is a convolutional neural network built by Oxford Universty. It , along with a family of VGG models, was entered in  ILSVRC-2014 and won first and second place for localization and classification. [1] It contains 19 layers of convolution, fully connected, and softmax layers. This model is going to be used as the basis for the fully convolutional network, as the paper uses. We'll load VGG19 by defining each layer of the network in code, and then we'll use a file that contains preloaded data to set the weights of the network. Afterwards we'll use transfer learning and create a new network that is called the deconvolution network. 

# Deconvolution Layer
The deconvolution layer, or more aptly named transpose convolution layer, takes input and performance transpose convolution. This can also be seen as convolution with a 1/k stride, where k is <=1. This essentially upsamples the input into a larger output. The goal of this layer is to construct the input as a semantically segmented image. In the paper by Long et all they use 3 transpose convolution layers, where two layers share similar shapes with earlier pooling layers in the VGG architecture, and the last layer outputs the final semantically segmented image. The first two layers get two pooling layers added to it. The reason for this is the input for the first deconvolution layer is very small, and upsampling that input will create a very coarse, blocky image. Earlier layers from VGG19 are fed to these two layers to help inject detail that was lost.


{% highlight python %}
def createConvolutionTransposeLayer(x_input, output_shape, weight_shape, stride=2):
    weights = tf.get_variable('weights', initializer = tf.truncated_normal(weight_shape, stddev=0.02))
    biases = tf.get_variable('biases', shape=[weight_shape[2]], initializer=tf.constant_initializer(0))

    convolution = tf.nn.conv2d_transpose(x_input, weights, output_shape, strides=[1,stride,stride,1], padding = 'SAME')
    return tf.nn.bias_add(convolution, biases)

{% endhighlight %}

The network consists of 3 transpose convolution layers, with the first two recieving skip connects from the 3rd and 4th pooling layer in VGG19. The final transpose convolution layer is the output layer and the final layer in the entire network. This layer will output the predicted label as an image of class based pixels.

{% highlight python %}
def createDeconvolutionNetwork(x_input, orig_input, pool_fuse1, pool_fuse2):
    
    with tf.variable_scope('dc1'):
        deconvolution_layer1 = createConvolutionTransposeLayer(x_input, tf.shape(pool_fuse1), [4, 4, pool_fuse1.get_shape()[3].value, num_classes], stride=2)
        print("dc1: " + str(deconvolution_layer1.get_shape()))
        #fuse layer with pool 4
        fuse_layer1 = tf.add(pool_fuse1, deconvolution_layer1)
        print("fuse1: " + str(fuse_layer1.get_shape()))
        
    with tf.variable_scope('dc2'):
        deconvolution_layer2 = createConvolutionTransposeLayer(fuse_layer1, tf.shape(pool_fuse2), [4, 4, pool_fuse2.get_shape()[3].value, pool_fuse1.get_shape()[3].value], stride=2)
        print("dc2: " + str(deconvolution_layer2.get_shape()))
        #fuse layer with pool 3
        fuse_layer2 = tf.add(pool_fuse2, deconvolution_layer2)
        print("fuse2: " + str(fuse_layer2.get_shape()))
        
    with tf.variable_scope('dc3'):
        #determine size
        shape1 = tf.shape(orig_input)
        shape2 = tf.pack([shape1[0], shape1[1], shape1[2], num_classes])
        #conv layer
        deconvolution_layer3 = createConvolutionTransposeLayer(fuse_layer2, shape2, [16, 16, num_classes, pool_fuse2.get_shape()[3].value], stride=8)
        print("output_layer: " + str(deconvolution_layer3.get_shape()))
        
    output = tf.argmax(deconvolution_layer3, dimension=3)
    
    return tf.expand_dims(output, dim=3), deconvolution_layer3
{% endhighlight %}

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
1. https://arxiv.org/pdf/1409.1556/
