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
![alt text](https://github.com/sdeck51/sdeck51.github.io/blob/master/images/inceptionArchitecture.png)
This is the actual model, which includes every layer that makes up inception v3, the connects to and from each layer, as well as the pretrained weights and biases. What we're going to do is load this using python and then have the ability to classify with the model.

##### cropped_panda.jpg
![alt text](https://github.com/sdeck51/sdeck51.github.io/blob/master/images/cropped_panda.jpg)
This is a pretty cute panda that's included to use as input for the model.

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
    
Each of these files are needed to be able to load up the Inception v3 model, and then classify and understand the output of classification.


#### Using Model to Classify 
- functions to do this

#### Transfer Learning
- setting up train/validation/test sets
- caching data
- build small network to attach

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


