---
layout: post
title: Tensorflow - Classification with Transfer Learning
---
<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>

0.869993329
0.868417501
0.87367034
0.883387983
0.882206023

0.908299208
0.908956587
0.909449637
0.909449637
0.903040349



This experiment will compare the results of transfer learning on AlexNet from Visualizing and Understanding Convolution Networks[cite]. The paper performs an experiment running AlexNet on Caltech-101 and Caltech-256. This will repeat the same experiment, only using GoogleNet instead.

For Caltech101 there are 102 classes with 9145 images in total. The experiment follows as taking 15 and 30 images from each class, training the network and then comparing the accuracy. This is done 5 times for each split and each set takes random training samples.




| # Train | Acc % 15/class | Acc % 30/class |
| - | - | - |
| (Bo et al. 2013) | - | 81.4 ± 0.33|
|(Jianchao et al., 2009) | 73.2 | 84.3 |
| (Zeiler et al., 2013) P | 83.8 ± 0.5 | 86.5 ± 0.5 |
| Inception v3 | 87.5 ± 0.7 | 90.7 ± 0.2|
{: .tablelines}


For Caltech256 there are 257 classes total with 30,607 images in total. As this set has more images per class the splits are increased from 15, and 30 to 45 and 60 in addition. Each split is ran 5 times with random training samples and the final accuracy is recorded.

| # Train | Acc % 15/class | Acc % 30/class | Acc % 45/class | Acc % 60/class |
| - | - | - |
|(Sohn et al., 2011)| 35.1  |   42.1    |   45.7    |   47.9    |
|(Bo et al., 2013)|40.5 ± 0.4|48.0 ± 0.2|51.9 ± 0.2|55.2 ± 0.3|
|(Zeiler et al, 2013)|65.7 ± 0.2|70.6 ± 0.2|72.7 ± 0.4|74.2 ± 0.3|
| Inception v3 | wait | wait |  wait | wait|
{: .tablelines}
