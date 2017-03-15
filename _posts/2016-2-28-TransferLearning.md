---
layout: post
title: Tensorflow - Transfer Learning
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

    ![alt text](https://github.com/sdeck51/sdeck51.github.io/blob/master/images/Forklift-Truck.jpg)
    0.9826 forklift
    0.0002 barbell
    0.0002 tractor
    0.0002 crane
    0.0001 golfcart

If you were able to get things working up to this point then congratulations! You have control of an extremely powerful image classifier. If you have problems that involve the classes this model was trained for then you're ready to go! This however isn't likely. You probably don't need to classify that x image was a forklift or y image was a panda. You probably have images in mind that you want to classify. What can we do about this? We have this large, powerful model that we would like to use but it doesn't classify what we want. Thankfully there's a method we can use to take advantage of this model and it's learned features, called Transfer Learning.

#### Transfer Learning
Transfer Learning is a method where we transfer what a model has learned into a new classifier, one which we define on our own. If we want to classify different types of birds then we know that this model will benefit that as it has trained on birds and has learned features from that. What we're going to do is remove the last fully connected layer and classification layer and create a new one. This process takes a few steps to optimally perform. We're dealing with a very large model and training it can take a lot of time if we're not clever about how to handle it.

What we'll be doing is in essence training a very small classification network. The input of this network however will be an output from the inception model. The inception model has so many generalized features that they can be used beyond the classes it was trained for, so we are going to have new data to classify, and classify quickly!

The input our network will be taking in will come from what is called the "bottleneck" layer. This layer comes right before the classification layer, so you can imagine we are just replacing classification layers. Putting images through the model up to this layer can take quite a bit of time for larger datasets. One of the datasets I classify took over 20 minutes to pass each image into the network. If we want to train our new network for several epochs(an epoch meaning every image has passed through the network) then this will take time. What we can do though is since passing each image through inception will yield the same output, we can run them through once, and then just use these "bottleneck values".



#### Training Modified Model
- Optimization, batching, etc

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


