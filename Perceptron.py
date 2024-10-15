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
        self.weights = [0] * input_size                 #Initialize weights to 0
        self.learning_rate = learning_rate              #set learning rate
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
        dot_product = 0     # Initialze dot_product
        
        print(f"\nPredicting for input vector: {x}")
        print(f"Current weights: {w}")
        
        for i in range(len(x)):                 # Loop through indices
            dot_product += x[i]*w[i]            # Multiply corresponding values
            
        print(f"Dot Product: {dot_product}")    # Display dot product before prediction

        if dot_product >= 0:                    # If DP meets threshold, return 1,
            print("Prediction: 1 (dot product >= 0)\n")
            return 1                            # Else return -1
        else:
            print("Prediction: -1 (dot product < 0)\n")
            return -1
        
    def train(self, input_vectors, targets, epochs=1):
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
        
    """
    Main program (main):
        Initialize a Perceptron object with the appropriate input_size and learning_rate
        Prepare the training set:
            Create binary vectors for the L and I characters
            Define their corresponding targets (for example L = -1, I = +1)
            Call the trian method to adjust the weights
            After training, call the evaluate method to test the perceptron on new data
    """

# Initialize the Perceptron model with 9 input features and a learning rate of 1
p = Perceptron(input_size=9, learning_rate=1)

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
x2 = [item for row in matrix_x2 for item in row]
input_vectors = [x1, x2]

# Display the flattened vectors for clarity
print("\nFlattened Vector form:\nx1", x1, "\nx2", x2)

# Train our perceptron model on the input vectors and targets :)
p.train(input_vectors, test_targets)










