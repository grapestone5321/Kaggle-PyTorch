# Kaggle-PyTorch

-------


# Pytorch
https://pytorch.org/

### SAVING AND LOADING MODELS
https://pytorch.org/tutorials/beginner/saving_loading_models.html

### ResNet in PyTorch - GitHub
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

### EfficientNet: Improving Accuracy and Efficiency through AutoML and Model Scaling
https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

## Paper:

### Deep Learning with PyTorch
https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf


-------

## PyTorch Tutorials - Complete Beginner Course
https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

### Python Engineer: Patrick Loeber
https://www.youtube.com/c/PythonEngineer/featured

Free Python and Machine Learning Tutorials!

Hi, I'm Patrick. I’m a passionate Software Engineer who loves Machine Learning, Computer Vision, and Data Science. 

I create free content in order to help more people get into those fields. 

If you have any questions, feedback, or comments, just shoot me a message! I am happy to talk to you :)

If you like my content, please subscribe to the channel!

Please check out my website for more information: https://www.python-engineer.com

If you find these videos useful and would like to support my work you can find me on Patreon: https://www.patreon.com/patrickloeber

### 1

5:45

PyTorch Tutorial 01 - Installation


### 2


18:28

PyTorch Tutorial 02 - Tensor Basics


### 3

15:54

PyTorch Tutorial 03 - Gradient Calculation With Autograd

### 4


13:13

PyTorch Tutorial 04 - Backpropagation - Theory With Example

### 5

17:31

PyTorch Tutorial 05 - Gradient Descent with Autograd and Backpropagation

### 6


14:16

PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer

### 7


12:11

PyTorch Tutorial 07 - Linear Regression

### 8

18:22

PyTorch Tutorial 08 - Logistic Regression

### 9

15:27

PyTorch Tutorial 09 - Dataset and DataLoader - Batch Training

### 10


10:43

PyTorch Tutorial 10 - Dataset Transforms

### 11


18:17

PyTorch Tutorial 11 - Softmax and Cross Entropy


### 12

10:00

PyTorch Tutorial 12 - Activation Functions


### 13


21:34

PyTorch Tutorial 13 - Feed-Forward Neural Network

### 14


22:07

PyTorch Tutorial 14 - Convolutional Neural Network (CNN)


### 15


14:55

PyTorch Tutorial 15 - Transfer Learning

### 16


25:41

PyTorch Tutorial 16 - How To Use The TensorBoard

### 17


18:24

PyTorch Tutorial 17 - Saving and Loading Models

### 18


41:52

Create & Deploy A Deep Learning App - PyTorch Model Deployment With Flask & Heroku


### 19

38:57

PyTorch RNN Tutorial - Name Classification Using A Recurrent Neural Net


### 20

15:52

PyTorch Tutorial - RNN & LSTM & GRU - Recurrent Neural Nets


### 21

28:02

PyTorch Lightning Tutorial - Lightweight PyTorch Wrapper For ML Researchers

### 22

13:29

PyTorch LR Scheduler - Adjust The Learning Rate For Better Results


-------

# Kaggle-PyTorch-Baseline



# PyTorch process

## 1 Settup Dependencies

## 2 Load Datasets


## 3 Define a Model (CNN)

### Model

### Parameters

## 4 Train the Model

### Save the model

## 5 Prediction

### load the model

### Test the model

## 6 Submit


# Issues
### --- Takes much time to compute (>9 hours)
## ideas:
-> W/ kaggle notebook is required in Testing/Prediction

-> Separate Training from Testing/Prediction

-> W/O kaggle notebook in Training


# numpy.arange
ref: https://numpy.org/doc/stable/reference/generated/numpy.arange.html

### numpy.arange([start, ]stop, [step, ]dtype=None)

Return evenly spaced values within a given interval.

Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop). For integer arguments the function is equivalent to the Python built-in range function, but returns an ndarray rather than a list.

When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use numpy.linspace for these cases.

## Parameters
    startnumber, optional
    Start of interval. The interval includes this value. The default start value is 0.

### stopnumber
    End of interval. 
    The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.

### stepnumber, optional
    Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. 
    If step is specified as a position argument, start must also be given.

### dtypedtype
    The type of the output array. If dtype is not given, infer the data type from the other input arguments.

## Returns

### arangendarray
    Array of evenly spaced values.

    For floating point arguments, the length of the result is ceil((stop - start)/step). 
    Because of floating point overflow, this rule may result in the last element of out being greater than stop.


## lyft competition
### using validate.zarr, at every : 
    valid_index = np.arange(0,len(valid_dataset),1000)


### for non-chopped data:  by hengck23
    local cv loss: 9.346676  (lb = 15.034)
    breakdown: 
    PERCEPTION_LABEL_CAR         9.259042 
    PERCEPTION_LABEL_CYCLIST    17.221859 
    PERCEPTION_LABEL_PEDESTRIAN  4.827463
    
    
    
