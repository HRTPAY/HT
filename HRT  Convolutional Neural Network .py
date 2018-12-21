HRT Wei
Classification of CIFAR-10 datasets is an open benchmark problem in machine learning. Its task is to classify a set of RGB images with size of 32 x32.
These images cover 10 categories: aircraft, cars, birds, cats, deer, dogs, frogs, horses, boats and trucks.
Wei
Target
The goal of this tutorial is to build a relatively small convolutional neural network for image recognition.
Focus on building a standardized network organization structure, training and evaluation
Provide an example for building larger and more complex models
# CIFAR-10 was chosen because it is complex enough to test most of the functionality in TensorFlow and can be extended to larger models
At the same time, because the model is small and the training speed is fast, it is more suitable for testing new ideas and testing new technologies.
Wei
Key Points of this tutorial
The # CIFAR-10 tutorial demonstrates several important aspects of building larger and more complex models on TensorFlow:
Core mathematical objects such as convolution, modified linear activation, maximum pooling and local response normalization
# Visualization of some network behaviors during training, including input image, loss, distribution and gradient of network behaviors
The calculating function of the moving average of the learning parameters of # algorithm and the use of these average values in the evaluation stage to improve the prediction performance
# implements a mechanism that decreases the learning rate over time
# Design a pre-access queue for input data, separating disk latency and high-overhead image preprocessing operations from the model for processing
We also provide a multi-GPU version of the model to show that:
# Can configure the model so that it can be trained in parallel on multiple GPUs
# Variable values can be shared and updated between multiple GPUs
Wei
# Model Architecture
6550
# These layers are eventually connected to the softmax classifier through full connection layer pairs
Wei
 CIFAR-10 Model
The CIFAR-10 model can maximize code reuse by constructing training diagrams with the following modules:
# model input: including inputs (), distorted_inputs (), etc., which are used to read CIFAR images and preprocess them, respectively, as input for subsequent evaluation and training.
# model prediction: including inference (), etc., for statistical calculation, such as classification of images provided
# model training: including loss () and train (), which is used to calculate losses, calculate gradients, and update variables to present the final results.
# Model Input
# model is input through input () and distorted_inputs () functions, which read files from CIFAR-10 binary files.
# Because the number of bytes stored per image is fixed, the tf.FixedLengthRecordReader function can be used.
The processing flow of # picture file is as follows:
Pictures are uniformly tailored to 24 x 24 pixels, and central areas are tailored for evaluation or random tailoring for training.
The # image will be whitened approximately, making the model insensitive to the dynamic range of the image.
For training, we use a series of other random changes to increase the size of data sets:
65
# Random variation of image brightness
Contrast of 65507
Wei
# Model Prediction
The prediction process of the # model is constructed by inference (), which adds the necessary operational steps to calculate the Logits of the predicted values. The corresponding model is organized as follows:
# Layer Name Description
# conv_1 implements convolution and rectified linear activation
# pool_1 Max pooling
Local response normalization of # norm_1
# conv_2 convolution and rectified linear activation
Normalization of # norm_2 Local Response
# pool_2 Max pooling
# local_3 Full Connection Layer Based on Modified Linear Activation
# local_4 Full Connection Layer Based on Modified Linear Activation
# softmax_linear performs a linear transformation to output Logits
Wei
# Model Training
65507
 Softmax regression adds a soft Max nonlinearity to the network output layer, and calculates the normalized prediction value and label's 1-hot encoding cross-entropy.
In the regularization process, we will apply weight attenuation loss to all learning variables. When the objective function of the model is used, the sum of the cross-entropy loss and the weight decay of ownership is obtained. The return value of the loss function is as follows.
The # train () function adds operations to minimize the objective function. These operations include calculating gradients, updating learning variables, and ultimately returning an operation step for performing all calculations on a batch of images.
