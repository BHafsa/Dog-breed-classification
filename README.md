[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview


The current project is concerned about detecting human and dog faces. When a dog is detected, his breed is also
identified using CNN models. The project combines a series of models designed to perform various tasks in 
a data processing pipeline.  



Several state-of-the-art classification CNN models are explored for breed classification. Only the best performing
model is saved to be used in real scenarios. 

## Dataset
Three different datasets were used during the current project:

1. [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
2. [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
3. [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)

## Requirements
The following packages are required to run the code in the current project:

```
pip install tensorflow
pip install matplotlib
pip install numpy
pip install pandas
pip install tqdm
pip install flask
```

## What's included

The project contains two parts. The first part is concerned about building suitable
models that will allow to achieve the overall goal of the project. The second part
focuses on deploying the models in a web application.

### 1. Model Development
During this first part, two detectors are used. The first one is based on cascade features and
open-cv2 to detect human faces. The second detector uses the ResNet pre-trained model
to detect the presence of dogs in an image. Finally, transfer learning is used to detect the breed of 
the detected dogs.

### 2. A flask application 
The second part of the project is concerned about providing a user-friendly tool
that allow to test the model and see how it is performing on custom images. The tool
is an interactive flask web application that allow to upload an image and detects 
the dogs and humans.



## Creator

**Bousbiat Hafsa**

- https://www.linkedin.com/in/hafsa-bousbiat-535ba6ba/
- https://github.com/BHafsa


## Copyright and license

Code released under the [MIT Licence](https://github.com/BHafsa/Dog-breed-classification/blob/main/LICENSE). 


