# Image Classifier

<h2>About</h2>

This project is an image classifier that is trained to distinguish between cats and dogs in images.

The purpose of this project is not to produce as optimized and computationally effective classifer as possible.
The main goal of the project is to get some expirience in building classifier and writing deep learning related Python code.

**LEARNING AND HAVING FUN**

<h2>Table of Content</h2>

1. [classifier.py](https://github.com/mishazakharov/ImageClassificator/blob/master/classifier.py)
    * This file contains the main code for image preprocessing and neural network construction-learning. 
    You can train a neural network on your own or use an existing version, the accuracy of which exceeds 91%!
    * I used feature extraction (transfer learning) from MobileNetV2!
    
2. [run.py](https://github.com/mishazakharov/ImageClassificator/blob/master/run.py)
    * This file runs the classification on the selected image. It should be run to correctly classify 
      the image.*WARNINGS are hided!*
      
3. [classi.py](https://github.com/mishazakharov/ImageClassificator/blob/master/classi.py)
    * Here is written the code for the convolutional neural network architecture under the name Alex-Net. 
      I do not have the necessary computing power to complete the training of this network, but 
      if you have one, you can try teaching it yourself!
      
4. [neural_net](https://github.com/mishazakharov/ImageClassificator/tree/master/neural_net)
    * This folder contains poorly trained ALEX-NET neural network.
    
5. [mobile_netv2](https://github.com/mishazakharov/ImageClassificator/tree/master/mobile_netv2)
    * This folder contains pre-trained mobile-netv2 neural network(feature extraction) with a
    flatten and dense layers on top of it! This one gets 91% accuracy on 100 valid cat-dog images.
    
6. [DOGCAT](https://github.com/mishazakharov/ImageClassificator/tree/master/DOGCAT)
    * This folder contains training and test images, that this neural network was trained and tested on!
    
    
    

