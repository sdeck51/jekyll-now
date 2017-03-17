---
layout: post
title: Tensorflow - Facial Feature Detector
---
![](http://i.imgur.com/90KjE6A.png?2)

In this post I go over how to make a facial feature detector. Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/7.%20FacialFeatureDetection_Tutorial/FaceDetector.ipynb).


# Purpose/Goal
The main goal for this tutorial is to demonstrate how one can build a facial feature detector from scratch using tensorflow. We'll go through several different models to demonstrate how one can make improvements that lead to an optimized model. We'll also talk about improvements that can be made.

# Data
The data for a feature detector is fairly important in defining the model we'll be building. Unlike image classification where you can simply assign a label to an image, feature detection needs to know where the features in the image are, using through coordinates. The data I'm using can be found [here](https://www.kaggle.com/c/facial-keypoints-detection/data). Labeled data consists of 7049 images, though many samples are missing certain labels. There are 30 unique values in a label, represented as an x and y coordinate for 15 features of the face. Below is an example of a image from the data set with the labels applied to the face.
<center>{% include image.html url="http://i.imgur.com/rPjZh9h.png"
description="2 dimensional convolution. [Feature extraction using convolution, Stanford]" size="250" %}</center>

The data is formatted in a csv file, where each row represents an image and it's labels. It's a fairly messy file so we need to work on extracting the image data along with the label data.

{% highlight python %}
def load():
    filename = testing_filename if isTesting else training_filename
    df = pd.read_csv(filename)  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

    df = df.dropna()  # drop all rows that have missing values in them.
    
    images = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    images = images.astype(np.float32)

    labels = df[df.columns[:-1]].values
    labels = (labels - 48) / 48  # scale target coordinates to [-1, 1]
    labels = labels.astype(np.float32)
    
    # zip data
    zipped_values = list(zip(images, labels))

    #shuffle the data
    random.shuffle(zipped_values)

    #unzip new shuffled data
    images, labels = zip(*zipped_values)
    
    return images, labels


# Wrapper for load() that reshapes the input to a image_size x image_size x channels
def loadData():
    x_image, y_label = load()
    x_image = x_image.reshape(-1, image_size, image_size, channels)
    return x_image, y_label
    
{% endhighlight %}

With the above code we can successfully load the data. We pull 

With this we need to simply split up the data in some fashion. In deep learning cross validation isn't as often used due to time. Deep networks can train from hours to days, weeks to months, and using cross validation isn't as helpful, along with the fact that cross validation is used when data is limited. In deep networks we use large datasets that (hopefully) span the set we'll test with.

# Building Neural Networks
For this tutorial we're going to build several different models. The reason for this is to demonstrate different technical techniques that can be used to optimize the model to create the best performing model. To make this easier to create and deplot we'll look at building some generating functions that we can call that will build the layers of the models that we will need. If you get lost refer to the [theory page](https://sdeck51.github.io/Convolutional_Neural_Network_Concepts/).

### Fully Connected Layer
Fully connected layers are the most basic layer. Every input weight and bias is connected up to every node, hence it's fully connected. 

{% highlight python %}
def createFullyConnectedLayer(x_input, width):
    # createFullyConnectedLayer generates a fully connected layer in the session graph
    # 
    # x_input - output from previous layer
    # width - width of the layer (eg for a 10 class output you need to end with a 10 width layer
    #
    # returns fully connected layer in graph
    #
    print("fc: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[x_input.get_shape()[1], width],
                             initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[width], initializer=tf.constant_initializer(0))
     
    matrix_multiply = tf.matmul(x_input, weights)
    
    return tf.nn.bias_add(matrix_multiply, biases)
{% endhighlight %}



Generally the layers are connected to an activation function. In our case we're looking at using linear rectifiers, as they train fast fast and are less prone to vanishing gradients than typical sigmoid and tanH functions.

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

# Building the Network
To begin I want to start by building a simple neural network to see how it fares. Afterwards I'll implement a larger convolutional neural network and implement concepts that have shown to boost performance of networks, namely:

To start off we're going to look at a very simple neural network and see what kind of results we can obtain from that. To do this we need to be able to build fully connected layers.

{% highlight python %}
def createFullyConnectedLayer(x_input, width):
    # createFullyConnectedLayer generates a fully connected layer in the session graph
    # 
    # x_input - output from previous layer
    # width - width of the layer (eg for a 10 class output you need to end with a 10 width layer
    #
    # returns fully connected layer in graph
    #
    print("fc: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[x_input.get_shape()[1], width],
                             initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[width], initializer=tf.constant_initializer(0))
     
    matrix_multiply = tf.matmul(x_input, weights)
    
    return tf.nn.bias_add(matrix_multiply, biases)
{% endhighlight %}

We'll need to be making multiple fully connected layers, as well as other types of layers, so I'll be making functions to generate these. The above will create a fully connected layer. Along with this we need to implement the activation function. In neural networks each node has some sort of saturating function, such as the sigmoind, or hyperbolic tangent. For deep networks, due to speed concerns as well as vanishing gradient issues, linear rectifiers are used. Also called ReLu, this function is simply the function x with a floor of zero.

Another concern we have is the weights need to be initialized in some fashion, because if they have the same initial values then they may end up learning similar features and create an underperforming model. People have found that simply applying gaussian a guassian distribution with zero variance works well for both improved training speed as well as preventing the networking from learning the same features, or even from it stop learning. We'll be using xavier initialization for this.

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

These two functions are all we need for now to build a simple model. We can create one more function that will be used to actually build the model.

{% highlight python %}
def createSimpleNetwork(x_input):
    with tf.variable_scope('in'):
        x_input = tf.reshape(x_input, [-1, image_size*image_size])
    with tf.variable_scope('hidden'):
        hidden_fully_connected_layer = createFullyConnectedLayer(x_input, 100)
        relu_layer = createLinearRectifier(hidden_fully_connected_layer)
    with tf.variable_scope('out'):
        return createFullyConnectedLayer(relu_layer, 30)
{% endhighlight %}

What we're doing here is creating a simple 2 layermodel in the tensorflow graph. It has a single hidden layer and output layer. Now I know this will not be a very good model, however we will see how certain techniques as well as building a deep convolutional neural network will be better.

With a model defined there is still additional work to be done. We need to define how to optimize the model.

### Optimization
Optimization is how a neural network learns. When you feed input to the model, it will also give an output. We're working with supervised learning, so when training we compare the output of a model with the actual label. If the prediction isn't a match with the label then we need to adjust the model. This is done by changing the weights and biases, which is done via optimization. I tried a few different optimizer and settled on stochastic gradient descent.

{% highlight python %}
graph = tf.Graph()

with graph.as_default():
    
    x_input = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 1])
    y_output = tf.placeholder(tf.float32, shape=[None, num_labels])
    is_training = tf.placeholder(tf.bool)

    #current_epoch = tf.Variable(0)  # count the number of epochs
    
    num_epochs=2000
    dropout = True
    global_step = tf.Variable(0, trainable=False)
    #learning rate
    learning_rate = 0.04
    
    #momentum
    momentum_rate = 0.9
    
    # get model
    prediction_output = createSimpleNetwork(x_input)
    
    loss_function = tf.reduce_mean(tf.square(prediction_output - y_output))
    #Momentumoptimizer implements SGD with momentum
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(loss_function, global_step = global_step)

{% endhighlight %}

To optimize we obviously need a cost function to optimize. For this problem we're dealing with error in distances of several points, so we're using mean squared error for the loss function. sum(1/2(x-x'))

### Training the network
The last large step we need to implement is the code to actually train the network.

{% highlight python %}
start = time.time()
train_loss_list = []
valid_loss_list = []
time_list = []
epoch_list = []
print("TRAINING: " + model_name)
with tf.Session(graph = graph) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for current_epoch in range(num_epochs+1):
        for batch_data, labels in batch(x_train, y_train, batch_size):
            feed_dict = {x_input: batch_data, y_output: labels, is_training: dropout}
            # training and optimizing
            session.run([optimizer], feed_dict = feed_dict)
            
        train_loss = lossFunction(get_predictions_in_batches(x_train, session), y_train)
        train_loss_list.append(train_loss)
        valid_loss = lossFunction(get_predictions_in_batches(x_validate, session), y_validate)
        valid_loss_list.append(valid_loss)
        #if(current_epoch % 10 == 0):
        # validate every so often

        current_time = time.time() - start

        hours, minutes, seconds = getTime(current_time)

        print("Epoch[%4d]" % current_epoch + "%d" % hours + ":%2d" % minutes + ":%2d " % seconds + "%f " % train_loss + " %f" % valid_loss + " %f " % learning_rate.eval() + "%f" % momentum_rate)

        time_list.append(current_time)
        epoch_list.append(current_epoch)
        # Evaluate on test dataset.
    test_loss = lossFunction(get_predictions_in_batches(x_validate, session), y_validate)
    print(" Test score: %.3f (loss = %.8f)" % (np.sqrt(test_loss) * 48.0, test_loss)) 
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    save_path = saver.save(session, model_path)
{% endhighlight %}



This code will start training your model. There is additional code beyond this that you will need to implement. Refer to the github page.

# Simple Neural Network Results

Once the model is finished running you can run predictions, as well as see how it learned over time. Below is a plot showing the training error vs the validation error.

<center>{% include image.html url="http://i.imgur.com/LzDXs5B.png"
description="Training and Validation over 1000 epochs." size="400" %}</center>

Running beyond doesn't show much improvement for the model, suggesting that the model cannot learn more. Along with this we can run some predictions. Since this isn't classification I'm going to display multiple images overlaid with their labeled feature locations and the models predicted feature locations.

<center>{% include image.html url="http://i.imgur.com/y0QDTlp.png"
description="Simple Neural Network Results" size="700" %}</center>

We can infer from the above image that the model is not sufficient. Some of the eye features are close, but the mouth features are structured correctly, but are extrememly off. With this we can actually take a few different avenues. If you want you can let it train longer. You can also modify the optimization method. For now though I want to build a convolutional neural network. Techniques I use on it will also be applied to this network to demonstrate their effectiveness.

# Convolutional Neural Network

*picture of model I'm building*
For this network we're going to be building a ConvNet roughly 10x larger than the simple neural network. I already introduced the fully connected layer and linear rectifier generating function. Now I need to introduce the convolution layer and pooling layer generating functions.

{% highlight python %}
def createConvolutionLayer(x_input, kernel_size, features, depth):
    # createConvolutionLayer generates a convolution layer in the session graph
    # by assigning weights, biases, convolution and relu function
    #
    # x_input - output from the previous layer
    # kernel_size - size of the feature kernels
    # depth - number of feature kernels
    #
    # returns convolution layer in graph
    #
    print("conv: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, features, depth],
                             initializer = tf.contrib.layers.xavier_initializer())
    
    biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0))
    
    convolution = tf.nn.conv2d(x_input, weights, strides=[1,1,1,1], padding='SAME')
    
    added = tf.nn.bias_add(convolution, biases)
    
    return tf.nn.relu(added)
{% endhighlight %}

Every convolution layer consists of weights and biases, organized in a way where weight connections are performing convolution on the input. Like the fully connected layer, the weights are applied with Xavier initialization.

{% highlight python %}
def createPoolingLayer(x_input, kernel_size):
    # createPoolingLayer generates a pooling layer in the session graph
    # 
    # The reason this exists is due to the last fully connected layer not needing a relu while others do
    # x_input - output from previous layer
    # kernel_size - size of the kernel
    #
    # returns pooling layer in graph
    # 
    print("pool: input size: " + str(x_input.get_shape()))
    return tf.nn.max_pool(x_input, ksize=[1, kernel_size, kernel_size, 1], strides=[1,kernel_size,kernel_size, 1], padding='SAME')
{% endhighlight %}

The pooling layer only needs to know about how much downsampling it will be performing, which is done through the kernel size.

With these new layer generating functions we can build the convolutional neural network.

{% highlight python %}
def createConvolutionalNetwork(x_input, isTraining):
    # Define convolution layers
    with tf.variable_scope('conv1'):
        convolution_layer1 = createConvolutionLayer(x_input, 3, 1, 32)
        pooling_layer1 = createPoolingLayer(convolution_layer1, 2)
    with tf.variable_scope('conv2'):
        convolution_layer2 = createConvolutionLayer(pooling_layer1, 2, 32, 64)
        pooling_layer2 = createPoolingLayer(convolution_layer2, 2)
    with tf.variable_scope('conv3'):
        convolution_layer3 = createConvolutionLayer(pooling_layer2, 2, 64, 128)
        pooling_layer3 = createPoolingLayer(convolution_layer3, 2)
        # Determine if used for training or test/validate. Only use dropout for training
        pooling_layer3 = tf.cond(isTraining, lambda: tf.nn.dropout(pooling_layer3, keep_prob=0.7), lambda: pooling_layer3)
    
    # Flatten output to connect to fully connected layers
    print("fc: input size before flattening: " + str(pooling_layer3.get_shape()))
    pooling_layer3_shape = pooling_layer3.get_shape().as_list()
    pooling_layer3_flattened = tf.reshape(pooling_layer3, [-1, pooling_layer3_shape[1] * pooling_layer3_shape[2] * pooling_layer3_shape[3]])
    
    # Define fully connected layers
    with tf.variable_scope('fc1'):
        fully_connected_layer1 = createFullyConnectedLayer(pooling_layer3_flattened, 500)
        fully_connected_relu1 = createLinearRectifier(fully_connected_layer1)
    with tf.variable_scope('fc2'):
        fully_connected_layer2 = createFullyConnectedLayer(fully_connected_relu1, 500)
        fully_connected_relu2 = createLinearRectifier(fully_connected_layer2)
    with tf.variable_scope('out'):
        output = createFullyConnectedLayer(fully_connected_relu2, num_labels)
        print("out: " + str(output.get_shape()))
    return output
{% endhighlight %}

With this new model function simply swap out createSimpleNetwork with createConvolutionalNetwork. Train the network with the same code. 

*Graph that has both NN and CNN*
*Faces with CNN*

Conv improved learning rate

One of the first improvements to suggest if you feel your model is training slowly is to simply increase the learning rate. This will increase the distance the optimization takes, which can increase the speed. It can also raise the chance that you won't converge. The issue is if you're optimizing to a minimum and your step size moves over it then you could end up oscillating past minimums, and also blow up.

*Graph that has both CNN02 and CNN04*
*Faces with CNN04*

As I just mentions, increasing the learning rate can be beneficial, but not when converging. Instead of having a constant learning rate, we can have a learning rate that is high at the beginning and that decays over time. This way we'll have small step sizes when we need to.

*Graph that has both CNN04 and CNNDecay*
*Faces with CNNDecay*

Once we're comfortable with our optimization changes we need to start thinking about other ways to increase performance of the model. One problem that most people come across is not having enough data. We can generate an infinite supply of data by augmenting our images. Obviously this isn't as good as having unique data but let's see how it improves our models.

### Data Augmentation

#### Flipping Images
#### Changing Brightness
#### Adding Noise

*Graph that has both CNNDecay and CNNDA*
*Faces with CNNDA*

Conv dropout

With data augmentation we should have a much better model. Even with this though we're having overfitting issues. The last change we're going to make to our model is implementing dropout. Dropout is a highly used technique that helps to generalize the model by disabling nodes while training. Instead of training all of the nodes in a layer, dropout disables randomly selected nodes(with a user defined ratio) and those nodes do not learn for that step. The idea is that this will discriminate features in images to certain nodes, so nodes in general will learn more unique features, rather than learning the same features from the same images. One disadvantage in using dropout is that training becomes a lot slower. This makes sense though as we're only training certain nodes on certain layers. What this gives us in return though is longer training with longer learning.

*Graph that has both CNNDecay and CNNDA*
*Faces with CNNDA*

# Results
![](http://i.imgur.com/qMv2z9k.png)

Acronyms in Model Name: 
- NN - Neural Network
- CNN - Convolutional Neural Network
- LRNx - Learning Rate of x
- EX - exponential decay of learning rate
- EP - Number of Epochs trained
- DA - Data Augmentation
- DRP - Dropout enabled

Current Best Model
![](http://i.imgur.com/y1EbHby.png)

Red points represent labels, while green points represent label predictions. You can see that the third row fourth column does not match up very well. We can see that the image is both off angle and the face is somewhat tilted.

####  Additional Improvements to try: 
- Additional data augmentation: Current augmentation includes horizontal flipping, brightness, and noise. Other augmentations such as rotating the face or cropping could produce a more robust model. This additional will need labels to be transformed in the same fashion.
- separate models for specific features: Instead of having one model for all of the labels, separate models can be used to calculate specific labels, such as each eye, eyebrow, nose. and mouth. 
- continued adjustments to optimizer: using different values for learning rate and momentum have shown changes to both the speed of training as well as overall error. 

## Things I want to add
- gif showing validation image predicted labels migrate while model is being trained
- Already have code working to do this. Will demonstrate the beginning training process using 1 batch steps.
