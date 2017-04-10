---
layout: post
title: Tensorflow - Classification with Transfer Learning
---
<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>

# Purpose
The purpose of this tutorial is to demonstrate how to classify images using Inception v3 from Google's Tensorflow library. It is also to demonstrate how to perform transfer learning on the model. Many large companies, from Google to Microsoft, have deep convolutional neural networks that they use for image classification. These networks are trained on the ILSVRC dataset from Imagenet[1]. This dataset consists of over 14 million images of which networks are trained on 1000 classes. These are great networks to learn how to use, but for the average user training one of these networks from scratch on their own isn't feasible. These networks take weeks and months to train using the dataset, sometimes on hardware made specifically for these tasks.An experiment is ran on Inception v3 following an experiment from Visualizing and Understand Convolution Networks[2].

# Data
The data required to run the experiments for this tutorial are the Caltech101Objects[3] and Caltech256Objects[4] datasets. The provided code however is created in a way such that you can import any jpeg images such that they are structured where the folder heirarchy is organized as classes, ie folder Horse has horse images in it.

# Inception v3
GoogleNet was the winner of 2014 ILSVRC[41 with a top-5 error of 6.67%. It is a 22 layer network that is detailed in the 2015 paper Going Deeper with Convolutions. Since its victory in the competition it has gone through different iterations. Inception v3 is a more recent model that we will be testing. There are newer version, a v4 as well as resnet version, that we are not looking at.  Google hosts the model online and it can be accessed [here] (http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz). The model comes with several files that need some explaining.

#### classification_image_graph_def.pb

<center>{% include image.html url="https://github.com/sdeck51/sdeck51.github.io/raw/master/images/inceptionArchitecture.png" description="GoogleNet Inception v3 . [5]" size="900" %}</center>
This is the model file that includes each layer, the connections to and from the layers, the weights of those connections, and the biases of each of the nodes. What we are going to do is load this into our program and classify with it.

#### cropped_panda.jpg

<center>{% include image.html url="https://github.com/sdeck51/sdeck51.github.io/raw/master/images/cropped_panda.jpg" description="GoogleNet Inception v3 . [5]" size="100" %}</center> This is a cropped panda image that is included to use for classification. One of the classes in the model is giant pandas, so  this image can be used to see if the classifier works.

#### imagenet_2012_challenge_label_map_proto.pbtxt

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
This includes all of the classes in the imagenet 2012 challenge. This file maps the class id and the human readable name. We are not going to understand when the model thinks the panda image is 89% class n01573342, so this will help us translate.

#### imagenet_synset_to_human_label_map.txt

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
        
Since inception v3 was trained on 1000 classes it has a 1000 vector output. This represents the probability of the input being each class. There are more than 1000 classes in the Imagenet dataset, so we need to know what each output represents in terms of output vector of the model and the class id it belongs to. This, paired with the mapping file above, will allow us to correctly state predictions in a human readable way.

With the files understood, building the network and mapping the labels can be performed. With the architecture inside the tensorflow graph classification can be ran. The model accepts JPEG images using the model, and with some helping functions a top x results can be displayed.

With the panda image we can run the network and view the top 5 results.

<center>{% include image.html url="https://github.com/sdeck51/sdeck51.github.io/raw/master/images/cropped_panda.jpg" description="Cropped panda" size="100" %}</center>

    0.8923 giant panda
    0.0086 indri
    0.0026 lesser panda
    0.0014 custard apple
    0.0011 earthsta

Any jpeg image can be ran on the network. Below is the results using a stegosaurus with top 2 results.


<center>{% include image.html url="http://i.imgur.com/v5jqKnV.jpg" description="Stegosaurus" size="200" %}</center>

        0.9998 triceratops
        0.0000 sunscreen

The model is extremely confident that it is a triceratops. This isn't the case of course, but the explanation as to why it thinks that is simple. It does not know what a stegosaurus is. There is no stegosaurus class in the ImageNet set, and so there's no reason for it to know what it is. This is somewhat limiting, as this network can only classify the 1000 classes it was trained on. In the case that someone wants to classify something outside of that class then the network will need to be modified.

## Transfer Learning

![alt text](https://github.com/sdeck51/sdeck51.github.io/raw/master/images/inception2.png)

Transfer Learning is a method where we transfer what a model has learned into a new classifier, one which we define on our own. We are interested in Inception v3's learned kernels. Since they can classify the 1000 classes it was given these kernels can most likely classify other classes. If we want to classify different types of birds then we know that this model will benefit that as it has trained on birds and has learned features from that. What we're going to do is create a new classification layer with the number of classes defined by us and attach it to inception v3. This process takes a few steps to optimally perform. Remember, we're dealing with a very large model and training it can take a lot of time if we're not clever about how to handle it.

What we'll be doing is in essence training a small classification network. The input of this network will not be images, but an intermediate staged output from the inception model. This intermediate stage is called the "bottleneck" layer. This layer comes right before the classification layer, so you can understand we're simply replacing the classification layer with one we defined. Putting images through the model up to this layer can take quite a bit of time for larger datasets(One set was roughly 30 minutes for one pass of every image). If we want to train our new network for several epochs(an epoch meaning every image has passed through the network) then this will take hours to run. One technique we can use though is caching. Since we are passing each image through Inception v3 several times it'll yield the same output. We're not touching the weights of the original network, so what we're going to do is pass each image through inception v3 once, extract the bottleneck output for each image, and then cache that data. Then when we start training out model, we'll use these new bottleneck values instead of the original images. Running Inception with Caltech256 for instance takes 30 minutes to get through one epoch. Caching the data reduces the training time dramatically.

With the new network created and bottleneck images cached, training can be performed on the new network. For loss I'm cross entropy and SGD with Momentum for the optimization. For a test I run the CalTech101Objects dataset split into 60% training, 40% validation/testing.

<center>{% include image.html url="http://i.imgur.com/59R4tai.png" description="Model loss" size="700" %}</center>
<center>{% include image.html url="http://i.imgur.com/uSmo8td.png" description="Model accuracy" size="700" %}</center>

The model ends up with an accuracy of 92.47% and loss of 0.3695. On top of the accuracy and loss I display a confusion matrix and random selection of predictions.

<center>{% include image.html url="http://i.imgur.com/1c7ngdw.jpg" description="Confusion Matrix for Caltech 101 Objects Dataset" size="900" %}</center>

<center>{% include image.html url="http://i.imgur.com/3TB7hB2.jpg" description="Model accuracy" size="800" %}</center>




## Experiment

This experiment will compare the results of transfer learning on AlexNet from Visualizing and Understanding Convolution Network[2]. The paper performs an experiment running AlexNet on Caltech-101 and Caltech-256. This will repeat the same experiment, only using GoogleNet instead.

For Caltech101 there are 102 classes with 9145 images in total. The experiment follows as taking 15 and 30 images from each class, training the network and then comparing the accuracy. This is done 5 times for each split and each set takes random training samples.




| # Train | Acc(%) 15/class | Acc(%) 30/class |
| - | - | - |
| (Bo et al. 2013) | - | 81.4 ± 0.33|
|(Jianchao et al., 2009) | 73.2 | 84.3 |
| (Zeiler et al., 2013) P | 83.8 ± 0.5 | 86.5 ± 0.5 |
| Inception v3 | 87.5 ± 0.5 | 89.8 ± 0.4|
| Inception v4 | 85.6 ± 0.6 | 88.5 ± 0.5|
| Inception ResNet v2 | 87.8 ± 0.8 | 90.1 ± 0.5|
{: .tablelines}


For Caltech256 there are 257 classes total with 30,607 images in total. As this set has more images per class the splits are increased from 15, and 30 to 45 and 60 in addition. Each split is ran 5 times with random training samples and the final accuracy is recorded.

| # Train | Acc(%) 15/class | Acc(%) 30/class | Acc(%) 45/class | Acc(%) 60/class |
| - | - | - |
|(Sohn et al., 2011)| 35.1  |   42.1    |   45.7    |   47.9    |
|(Bo et al., 2013)|40.5 ± 0.4|48.0 ± 0.2|51.9 ± 0.2|55.2 ± 0.3|
|(Zeiler et al, 2013)|65.7 ± 0.2|70.6 ± 0.2|72.7 ± 0.4|74.2 ± 0.3|
| Inception v3 | 79.4 ± 0.4 | 81.9 ± 0.3 |  83.0 ± 0.2 | 84.3 ± 0.5 |
| Inception v4 | 80.1 ± 0.6 | 82.5 ± 0.4 |  85.2 ± 0.5 | 86.5 ± 0.4 |
| Inception ResNet v2 | 81.0 ± 0.8 | 84.4 ± 0.3 |  86.2 ± 0.2 | 86.7 ± 0.7 |
{: .tablelines}

## Full Code
https://github.com/sdeck51/CNNTutorials/blob/master/5.%20TransferLearning_Tutorial/TransferLearning.ipynb

## References
1. Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV,2015.

2. M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. CoRR, abs/1311.2901v3, 2013.

3. L. Fei-Fei, R. Fergus and P. Perona. One-Shot learning of object categories. IEEE Trans. Pattern Recognition and Machine Intelligence. In press.

4. Griffin, G. Holub, AD. Perona, P. The Caltech 256. Caltech Technical Report

5. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015

