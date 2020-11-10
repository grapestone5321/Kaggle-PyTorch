# Kaggle-PyTorch-Baseline

## Udacity-deep-learning-v2-pytorch/intro-to-pytorch/

: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch

### Deep Learning with PyTorch
This repo contains notebooks and related code for Udacity's Deep Learning with PyTorch lesson. This lesson appears in our AI Programming with Python Nanodegree program.

- Part 1: Introduction to PyTorch and using tensors
- Part 2: Building fully-connected neural networks with PyTorch
- Part 3: How to train a fully-connected network with backpropagation on MNIST
- Part 4: Exercise - train a neural network on Fashion-MNIST
- Part 5: Using a trained network for making predictions and validating networks
- Part 6: How to save and load trained models
- Part 7: Load image data with torchvision, also data augmentation
- Part 8: Use transfer learning to train a state-of-the-art image classifier for dogs and cats






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
    
    
    
