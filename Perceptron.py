# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:29:33 2024

@author(s): Cole Ralph, Austin Snyder, Bobi Vladimirov 
        Central Washington University
"""
"""
Class Perceptron:
    define class with:
        list of weights
        learning rate
        Methods for initialization, prediction, training, and evaluation
"""
class Perceptron:
    #initialization method (__init__)
    def __init__(self, input_size, learning_rate=1): #Input size (number of features in the input), learning rate
        self.weights = [0] * input_size   #Initialize weights to 0
        self.learning_rate = learning_rate  #set learning rate
"""
Predition method (predict):
    Input: a single input vector x (so matrix must be flattened to 1D vector)
    Compute dot product of weights and vector x
    Apply the activation function (sign function):
        If the dot product is greater than or equal to 0, return +1
        Else, return -1
"""

"""
Training method (train):
    Input training_inputs (a list of input vectors), targets (a list of expected outputs)
    For each input and target in the dataset
        Use the predict method to compute the output (y)
        Compare the predicted output y to with the target (t)
        If the output is wrong, update the weights:
            For each weight w_i:
                Update w_i = w_i + learning_rate * (t - y) * x_i
    Repeat for a number of iterations until the model converges
"""

"""
Evaluation method (evaluate):
    Input: (test_inputs, test_targets)
    For each input in the test set, use the predict method
    Compare the predicted output with the true target
    Compute the accuracy (the percentage of correct prediction)
"""

"""
Main program (main):
    Initialize a Perceptron object with the appropriate input_size and learning_rate
    Prepare the training set:
        Create binary vectors for the L and I characters
        Define their corresponding targets (for example L = -1, I = +1)
    Call the trian method to adjust the weights
    After training, call the evaluate method to test the perceptron on new data
"""
            