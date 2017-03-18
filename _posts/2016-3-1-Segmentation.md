---
layout: post
title: Tensorflow - Segmentation
---

In this post I'll show how you can perform segmentation using CNNs! Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/6.%20Segmentation_Tutorial/Segmentation2.ipynb)

# Intro
For this tutorial I'm following the paper [here](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). Fair warning, if you want to run this you'll need a gpu with at least 8GB, otherwise you'll have to use the cpu version which will take eons to finish(I'm running a 12 hour training test atm and will probably have a terrible model).

![](http://i.imgur.com/ysw2ZZx.png?1)

# Purpose/Goal
The purpose of this tutorial is to demonstrate how to do per pixel classification using transpose convolution layers on a deep network. This process creates segmented images that can separate the classe object in a scene.





# Process

### Things to discuss
- Loading the data
- Building the Model (VGG 16/19)
- Loading the weights
- Implementing new classifier via transfer learning
- Transpose Convolution Layers - new layer type
- Fuse Layers/skip connections - adding previous layers to future layers

### What is Image Segmentation?
![](http://i.imgur.com/mSJDVCS.jpg)![](http://i.imgur.com/qZh484g.png)
In computer vision, image segmentation is the idea of partitioning an image into segments. These segments represent objects and boundaries that can be used to more easily label or classify what is in an image.

### Fully Convolutional Networks
Fully Convolutional Networks(FCN) are fairly new architectures [CITE]. Like Convnets, FCNs impose convolution and pooling layers to extract feature maps from input images. What differs FCNs from traditional classification ConvNets is instead of classifying entire images FCNs classify every pixel. There are no fully connected layers, which is how ConvNets  and there are transposed convolution layers, used for upsampling. This upsampling layer/s is/are also learned. Additionally skips connections are used in various layers towards the upsampling layer to hopefully capture finer grain features in the image. In this tutorial we'll attempt to build a FCN using the popular VGG16 model.

# Data
The data we'll be using is from the MIT Scene Parsing website [here](http://sceneparsing.csail.mit.edu/). It contains 20,000 training images, and 2000 validation images across 151 different classes. The data we need is simply formatted in 4 folders that contain training images, training labels, validation images, validation labels. Due to having so many images it's a good idea to cache them on disc for quicker access.

The first thing we need to do is collect the names of the 
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

This will allows us to then grab each image

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

After 5-10 minutes or so the data should be fully loaded. Before moving on I found it to be a good idea to cache the files in a pickle file. I've had Jupyter Notebook crash which caused me to have to repull the data and waiting each time was aggravating.

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


# Building the Model

If you've gone through my Facial Feature Detector tutorial then this will be fairly similar. The main difference is that instead of building a custom model we're going to mimic the VGG 19 model. The reason we're doing this is because the paper mentioned at the beginning listed a few different networks that were used, which includes VGG 19, so I wanted to use the larger one. We also have access online to pretrained weights so we won't have to train the network from scratch.

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

![](http://imgur.com/AyWvWf6.png)

![](http://imgur.com/BhJCNm8.png)

![](http://imgur.com/1XUpg8k.png)

I want to show a few train/validation predictions over the course over training the network. Also recording loss over time.
