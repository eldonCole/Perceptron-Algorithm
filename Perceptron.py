import numpy as np
import util
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:29:33 2024

@author(s): Cole Ralph, Austin Snyder, Bobi Vladimirov 
        Central Washington University
"""


#Define class Perceptron
class Perceptron:
    """
    Perceptron Class:
        A single-layer perceptron for binary classification tasks.

    Attributes:
        weights : list
            A list representing the weights of the perceptron model.
    learning_rate : float
        Learning rate for adjusting the weights during training.
    """
    def __init__(self, input_size, learning_rate=1):    #Input size (number of features in the input), learning rate
        """
        Initializes the Perceptron with the given input size and learning rate.
        
        Parameters:
        input_size : int
            Number of features in the input.
        learning_rate : float, optional
            Learning rate for training (default is 1).
        """
        self.weights = np.random.rand(input_size + 1).tolist()    #Initialize weights randomly (0,1) (include one for bias)
        self.learning_rate = learning_rate                        #set learning rate
        print("\nInitialized weights:", self.weights)

    def predict(self, input_vector):
        """
        Predicts the output (-1 or 1) for a given input vector based on the current weights.

        Parameters:
        input_vector : list
            A list of input values (1D vector) representing features.

        Returns:
            int
                1 if the dot product of input vector and weights is >= 0, otherwise -1.
        """
        x = input_vector    # Alias x for vector
        w = self.weights    # Alias w for weights
        dot_product = np.dot(x, w) # Dot product of vector and weights
        
        print(f"\nPredicting for input vector: {x}")
        print(f"Current weights: {w}")
            
        print(f"Dot Product: {dot_product}")    # Display dot product before prediction

        return util.bipolar_activation(dot_product)

    def train(self, input_vectors, targets, epochs=50):
        """
        Trains the perceptron model using the provided input vectors and targets over a number of epochs.

        Parameters:
        input_vectors : list of lists
            A list where each element is a list representing a flattened input vector (e.g., the binary representations of 'L' or 'I').

        targets : list
            A list of expected outputs corresponding to each input vector. Typically, -1 or 1, where each represents the expected class label.

        epochs : int, optional (default=1)
            The number of times to iterate over the entire training set. Each epoch performs a complete pass through all input-target pairs.

        Process:
            - For each epoch, the model iterates over each input vector and its corresponding target.
            - It uses the `predict` method to calculate the current prediction `y`.
            - The difference between the prediction and the target (error) is used to update the model's weights based on the perceptron learning rule.
            - Weights are adjusted iteratively to minimize prediction error over time.
        """
        for epoch in range(epochs): #repeat for as many epochs as we choose
            print(f"\nEpoch {epoch + 1}")
            for x, t in zip(input_vectors,targets): # Loop through the input and targets as tuples, so the model differentiates them
                print(f"\nTraining on input vector: {x}, target: {t}")
                y = self.predict(x) # Predict the output
                print(f"Prediction: {y}, Target: {t}, Error (t - y): {t - y}")
                for i in range(len(self.weights)): # Loop through each of the weights indices
                    self.weights[i] += self.learning_rate * (t-y) * x[i]    # Apply the perceptron learning rule ;)
                print(f"Updated weights: {self.weights}")

    """
    Evaluation method (evaluate):
        Input: (test_inputs, test_targets)
        For each input in the test set, use the predict method
        Compare the predicted output with the true target
        Compute the accuracy (the percentage of correct prediction)
    """


# Initialize the Perceptron model with 9 input features and a learning rate of 1
p = Perceptron(input_size=9, learning_rate=0.01)

# Define two 3x3 binary matrices representing the letter L (x1) and the letter I (x2)
matrix_x1 = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 1]
]

matrix_x2 = [
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1]
]

# Some I's
#010010010;111010111;110010111;011010111;111010011;111010110;100100100;001001001;011001011;110100110;100100000;000100100;010010000;000010010;001001000;000001001;

# Define the expected targets for each input vector: -1 for L, 1 for I
test_targets = [-1, 1]  # -1 for L, 1 for I

# Display input matrices for clarity
print("\nInput Matrix x1:\n")
for row in matrix_x1:
    print(row)

print("\nInput Matrix x2:\n")
for row in matrix_x2:
    print(row)

# Flatten each matrix into a 1D vector as required by the perceptron
x1 = [item for row in matrix_x1 for item in row]
x1.append(-1)   # bias
x2 = [item for row in matrix_x2 for item in row]
input_vectors = [x1, x2]
x2.append(-1)   # bias

# Display the flattened vectors for clarity
print("\nFlattened Vector form:\nx1", x1, "\nx2", x2)

# Train our perceptron model on the input vectors and targets :)
p.train(input_vectors, test_targets)










