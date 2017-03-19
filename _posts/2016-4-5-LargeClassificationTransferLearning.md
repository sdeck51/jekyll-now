---
layout: post
title: Tensorflow - Classification with Transfer Learning
---

In this post I'll go over how you can harness a deep neural network and classify your own images quickly!
Full code [here](https://github.com/sdeck51/CNNTutorials/blob/master/5.%20TransferLearning_Tutorial/TransferLearningWithInception.ipynb)

# Intro
Here we learn how we can transfer learned features from one dataset to another.

# Purpose/Goal
The purpose of this tutorial is to introduce the concept of transfer learning. Transfer learning is a powerful tool in taking advantage of very deep networks trained on millions of images and using it for ones own classification. The problem with using a deep network on your own is that fully training the system can takes weeks. Using a network that is pretrained is one solution, however that network is made to classify the data it trained on. What if you have a set of classes you want to classify, and you need a large network to do it? You could train from scratch, but you could also use transfer learning and take advantage of a pretrained network. Using transfer learning will allow you to quickly train your data.

# Data
For the data in this tutorial we'll be looking at datasets using JPEG files. One problem I have training models is sometimes half of the work is just getting the data into your program. For this we'll implement a simple interface where data will be accessed in folder that will define the class of the image. So for example if you have a lot of images of planes you'll place them in a folder marked planes. Then the code will use the folder name as the class name. 

I'm going to demonstrate transfer learning using a few different datasets. In one we have 600 images of 6 different types of birds. Another we'll see what kind of accuracy we can get classifying male and female faces using a cropped celebrity database. The last one has 257 classes of random objects. 

# Process
### Things to discuss
#### Loading a Model
- download/loading
- loading weights

The first thing we need to do is get access a large, pretrained CNN model. There are various models available online, and for this tutorial I'll demonstrate how to access the Inception V3 model from Google. Google makes it fairly easy to download and use their model. If you wish to manually access it you can simply go [here](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz). Otherwise there is various code for downloading(as well as extracting) tar zipped files.
{% highlight python %}
def download(url, directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = url.split('/')[-1]
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
    
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                           (filename,
                            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
    
        filepath, _ = urllib.request.urlretrieve(url,filepath,  _progress)
        print("")
        statinfo = os.stat(filepath)
    else:
        print('File already downloaded')
    tarfile.open(filepath, 'r:gz').extractall(directory)

{% endhighlight %}

Now what is included in this zipped file is a little complicated. Let's go through each file

##### classification_image_graph_def.pd

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/inceptionArchitecture.png)
This is the actual model, which includes every layer that makes up inception v3, the connects to and from each layer, as well as the pretrained weights and biases. What we're going to do is load this using python and then have the ability to classify with the model.

##### cropped_panda.jpg

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/cropped_panda.jpg)
This is a pretty cute panda that's included to use for testing classification. Possibly the most important file.

##### imagenet_2012_challenge_label_map_proto.pbtxt

    ...
    n02869249	bones, castanets, clappers, finger cymbals
    n02869563	boneshaker
    n02869737	bongo, bongo drum
    n02869837	bonnet, poke bonnet
    n02870526	book
    n02870676	book bag
    n02870772	bookbindery
    n02870880	bookcase
    n02871005	bookend
    n02871147	bookmark, bookmarker
    n02871314	bookmobile
    n02871439	bookshelf
    n02871525	bookshop, bookstore, bookstall
    ...

Each class that was used for training in inception v3 has a class value as well as class id. This file maps the class id to the class value.

##### imagenet_synset_to_human_label_map

    ...
    entry {
      target_class: 391
      target_class_string: "n01558993"
    }
    entry {
      target_class: 392
      target_class_string: "n01560419"
    }
    entry {
      target_class: 393
      target_class_string: "n01580077"
    }
    ...

Each class also should have a name for us humans to easily understand what the classes are. This file maps the class id to a class name.
    
Each of these files are needed to be able to load up the Inception v3 model, and then classify and understand the output of classification. From this we can parse the files to get class values and class ids from the dataset that was used to train Inception v3. Look at the code for more detail on how this is done.

### Building the Model

After we have an understanding of the files we can now build the model as a graph in tensorflow. This is fairly simple as a user to do as the tensorflow api is native with the model file's .pd format.

{% highlight python %}
graph = tf.Graph()

with graph.as_default():

    path = os.path.join(model_directory, map_graph_def)
    with tf.gfile.FastGFile(path, 'rb') as file:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(file.read())

        #remember to set name to empty string else it doesn't work
        tf.import_graph_def(graph_def, name='')
{% endhighlight %}

With this we have a fulling loaded Inception v3 model in tensorflow! Now we need to know how to use it to classify images. For this I've made a few python functions to classify jpeg images.

### Making a Prediction
To make a prediction with our model we need to first make sure that the tensorflow session is running, and that the graph that is in the session is the one we set up with our model.

{% highlight python %}
# Create a TensorFlow session for executing the graph.
session = tf.Session(graph=graph)
{% endhighlight %}

It's as simple as that! Now that the graph is up and ready to be ran we can feed an image into it to classify it. To classify something we need to define what input we want to feed the session as well as what output we want in return. For this I have a simple function.

{% highlight python %}
def predictClass(image_path):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        
    prediction = session.run(y_output, feed_dict={jpeg_tensor_name: image_data})

    return np.squeeze(prediction)
{% endhighlight %}

We first load the image from it's image path and then using session.run we define the tensor we want returned to us, as well as the input tensor and input data we want through the feed dictionary. What this will return is a vector/array of 1000 elements, where each element represents the probability the input image was its class. What I do is instead of just finding the top class I grab the top x classes. You'll see in various papers that CNN models aren't generally rated by their top guess, but their top 5 guesses. So when you see the term top-5 error it means the model error for predictions where what it predicted didn't fall in the top 5 highest predictions. So for this we're going to simply reorder the predictions to see what the top x are, and then using the mapping we can output the class name along with it's probability.

{% highlight python %}
def calculateScore(predictions, amount):
    #sorting the predictions
    prediction_args_sorted = predictions.argsort()
    prediction_percent = np.sort(predictions)[:amount:-1]
    topWinners = prediction_args_sorted[::-1]
    i=0
    for winner in topWinners[:amount]:
        uid = class_to_id[winner]
        name = uid_to_name[uid].split(",")[0]
        print('%.4f' % prediction_percent[i] + " " + name)
        i+=1
{% endhighlight %}

{% highlight python %}
def classify(image_path, amount):
    image_path = os.path.join(model_directory, image_path)
    display(Image(image_path))
    calculateScore(predictClass(image_path), amount)
{% endhighlight %}

We can use the classify function to now classify JPEG images using the model we've loaded. All you need is to pass it the image path as well as an integer representing how many predictions you want back. Eg if I want the top 5 predictions I'll pass it a 5.

Let's use the given panda image and see what the model classifies it as

{% highlight python %}
classify('cropped_panda.jpg', 5)
{% endhighlight %}

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/cropped_panda.jpg)

    0.8923 giant panda
    0.0086 indri
    0.0026 lesser panda
    0.0014 custard apple
    0.0011 earthstar

So it can classify a panda, but what else? From here find any image you want and just set the path in the function. Here's it's take on a forklift.

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/Forklift-Truck.jpg)

    0.9826 forklift
    0.0002 barbell
    0.0002 tractor
    0.0002 crane
    0.0001 golfcart

If you were able to get things working up to this point then congratulations! You have control of an extremely powerful image classifier. If you have problems that involve the classes this model was trained for then you're ready to go! This however isn't likely. You probably don't need to classify that x image was a forklift or y image was a panda. You probably have images in mind that you want to classify. What can we do about this? We have this large, powerful model that we would like to use but it doesn't classify what we want. Thankfully there's a method we can use to take advantage of this model and it's learned features, called Transfer Learning.

## Transfer Learning

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/inception2.png)

Transfer Learning is a method where we transfer what a model has learned into a new classifier, one which we define on our own. If we want to classify different types of birds then we know that this model will benefit that as it has trained on birds and has learned features from that. What we're going to do is remove the last fully connected layer and classification layer and create a new one. This process takes a few steps to optimally perform. We're dealing with a very large model and training it can take a lot of time if we're not clever about how to handle it.

What we'll be doing is in essence training a very small classification network. The input of this network however will be an output from the inception model. The inception model has so many generalized features that they can be used beyond the classes it was trained for, so we are going to have new data to classify, and classify quickly!

The input our network will be taking in will come from what is called the "bottleneck" layer. This layer comes right before the classification layer, so you can imagine we are just replacing classification layers. Putting images through the model up to this layer can take quite a bit of time for larger datasets. One of the datasets I classify took over 20 minutes to pass each image into the network. If we want to train our new network for several epochs(an epoch meaning every image has passed through the network) then this will take time. What we can do though is since passing each image through inception will yield the same output, we can run them through once, and then just use these "bottleneck values".

To obtain bottleneck values we need to have access to the bottleneck tensor in the model. With this tensor we can run a session where we return the output of the bottleneck tensor.

{% highlight python %}
def getBottleneckValues(image_path):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = session.run(bottleneck_layer, feed_dict={jpeg_tensor_name: image_data})

    return np.squeeze(bottleneck_values)
{% endhighlight %}



Because this process can take a while to run, it's a good idea to cache the values onto disk so if you want to use it later you can just pull the values from a file rather than recalculating the values. For caching I use the pickle module and 

{% highlight python %}
def bottleneckCache(cache_path, images=None, image_paths=None):

    if os.path.exists(cache_path):
    # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            bottleneck_values = pickle.load(file)

        print("Data loaded: " + cache_path)
    else:
        # The cache-file does not exist.

        # Call the function / class-init with the supplied arguments.
        bottleneck_values = processImages(images=images, image_paths=image_paths)
        try:
        # Save the data to a cache-file.
            with open(cache_path, mode='wb') as file:
                pickle.dump(bottleneck_values, file)
        except EOFError:
            return {}
        print("Data saved: " + cache_path)

    return bottleneck_values
{% endhighlight %}




### Separate the Data

Once the values are cached we can set up the data for training/validation/testing. You can use any method you're used to. I simply randomize the order of the data and split it up into training and validation samples.

### Create new Classification Layer

Once our data is set up we'll define the new classiciation layer. This will take in the bottleneck values and output predictions. To do this we're going top create a fully connected layer that has a shape of the number in input bottleneck values by the number of output values which will be the number of new classes. We also give biases which equal to the number of classes. The input of the layer multiplies with the weights and the biases are added to that. This is then put through a softmax classification layer to output prediction percent per class.

Along with the newly added network we also need to define the cost function. This is the function that will be optimized to build the best fit model. For multi class classification a popular method to use is cross entropy. When we get an output from our model we'll have percentages for each class indicating it's that percentage.

For example

    dog 50%
    cat 22%
    mouse 18%
    lizard 10%

In the above case lets say that the input was in fact a dog. While the output has the top class as a dog we would like that value to be closer to 100%. At the same time let's look at this scenario.

    dog 95%
    cat 4%
    mouse 1%
    lizard 0%
    
Dog is clearly the choice here, though it still isn't 100%. We don't want to penalize the model as much for being 5% off, while we would want to penalize it more for being "more" incorrect.

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/crossentropy.PNG)

Since we are working with values between 0 and 1, you can see why we need to have the negative value. The summation is used to simply get the single class that that image actually belongs to and take the log of the predicted percentage. If this percentage is closer to 1, the output of the function will be lower, while as it goes farther away is gets higher.

### Optimization

With the cost function undertood we can apply an optimizer. The optimizer is what makes the model learn. Optimizing adjusts the weights and biases of the network to be closer to the actual labels. When you feed an image to the model, it'll output it's prediction. The learning process is simply changing those weights/biases so if we were to feed the image to the model again that the output prediction would be closer to what we want. A popular optimizer is the simple stochastic gradient descent. Gradient Descent is an optimization method that involves following the gradient of a function to reach a minimum. Stochastic gradient descent follows the same algorithm, however in SGD we can "batch" the input and optimize multiple inputs at once. This improves the speed of the algorithm by taking single steps for multiple inputs. Along with this you can define the step size that the optimizer will take. If it's too large then it may pass over minimums in the model's cost function and never converge. On the other hand if it's too small it may simply take too long to reach a minimum. Once method of dealing with this is to introduce a exponentially decaying step length, so at the beginning of training we'll have x step size, and over time that length will exponentially decay.



{% highlight python %}
with graph.as_default():
    
    x_input = tf.placeholder(tf.float32, shape=[None, bottleneck_length])
    y_output = tf.placeholder(tf.float32, shape=[None, num_classes])
    y_output_class = tf.argmax(y_output, dimension=1)

    num_epochs=2000
    dropout = True
    global_step = tf.Variable(0, trainable=False)
    #learning rate
    learning_rate = tf.train.exponential_decay(.01, global_step, num_epochs, 0.90, staircase = True)
    
    with tf.variable_scope('newfc'):
        weights = tf.get_variable('weights', shape=[bottleneck_length, num_classes],
                                 initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[num_classes], initializer=tf.constant_initializer(0))

        matrix_multiply = tf.matmul(x_input, weights)

        y_out = tf.nn.bias_add(matrix_multiply, biases)
        
    y_class = tf.nn.softmax(logits=y_out)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_output, logits=y_out)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_output,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)
   
{% endhighlight %}

### Training

With everything set up we can finally begin training our model. To do this we need to do a few things. Firstly we'll want to implement a batching scheme that will take batches of our training data and feed it into the model. This will speed up the model by optimizing multiple inputs at a time. We also need to define how long to train our network. Lastly we'll want to see how the model is improving by calculating the training and validation accuracy.

Code for that stuff



### Results
After a model has been trained we can see how well it works! The main method of seeing how it runs is to determine its error, or inversely it's accuracy. For large models a lot of groups like to check the top-5 error, so we'll want to take in the top 5 predictions as we did when originally classifying the network, though this time by feeding the model the bottleneck values. If you separated your data with a test set this is where you should use it. Simply classify your test data and return both top1 and top5 results.

Another method for seeing how well a model is working is by generating a confusion matrix. A confusion matrix lists the classes of a model and the predicted class of a model. Below is an example demonstrated using a small bird dataset.

*image of thing*

Along with this I like to display images from the set along with their actual labels and predicted labels.
*image of thing*
#### Results
- validation accuracy
- predictions
- confusion matrix

# Showing examples using 3 datasets
One thing I want to demonstrate is showing the 6 class bird dataset, where some of the classes have already been trained on while the others haven't.

Case with 257 class data

Case with faces.

Have confusion matrices for all. Overall accuracy.


### Progress
Code works! Just need to format results.

# Results
I have three different data sets. I need to upload % accuracy. Would be good to pick out a random batch and show their classification.

![](http://i.imgur.com/57cNk4l.jpg)
![](http://i.imgur.com/4WFTmcx.jpg)
![](http://i.imgur.com/NAy6MWl.png)


