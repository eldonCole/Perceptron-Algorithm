import random
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
    def __init__(self, input_size, learning_rate=1.0, a_function=1):    #Input size (number of features in the input), learning rate
        """
        Initializes the Perceptron with the given input size and learning rate.
        
        Parameters:
        input_size : int
            Number of features in the input.
        learning_rate : float, optional
            Learning rate for training (default is 1).
        """
        self.weights = np.random.rand(input_size).tolist()    # Initialize weights randomly (0,1) (include one for bias)
        self.learning_rate = learning_rate                        # set learning rate
        self.a_function = a_function                              # Bipolar = -1, Unipolar = 1

        if a_function != 1 and a_function != -1:
            raise ValueError("Invalid activation function: Bipolar = -1, Unipolar = 1")

        #print("\nInitialized weights:", self.weights)

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
        
        #print(f"\nPredicting for input vector: {x}")
        #print(f"Current weights: {w}")
            
        #print(f"Dot Product: {dot_product}")    # Display dot product before prediction

        # Activation function: Bipolar = -1, Unipolar = 1
        if self.a_function == 1:
            return util.unipolar_activation(dot_product)

        return util.bipolar_activation(dot_product)

    '''
    Inputs:

        List of vectors (or tuples): this is the test set with both L's and I's

        HashSet<tuple, int> : this is the way to identify which class a test point is in.

        Returns: float of percent right, (FUTURE ADD CONFUSION MATRIX)
    '''

    def evaluate(self, test_set, class_map):
        correct = 0

        for i in test_set:
            if self.predict(i) == class_map.get(i):
                correct += 1

        percent = correct / len(test_set) * 100
        # Return a confusion matrix, along with percent right
        return percent

    def train(self, input_vectors, class_set, epochs=30):
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
            #print(f"\nEpoch {epoch + 1}")
            for x in input_vectors: # Loop through the input and targets as tuples, so the model differentiates them
                t = class_set.get(x)  # this is the target for the current tuple (vector)
                #print(f"\nTraining on input vector: {x}, target: {t}")
                y = self.predict(x) # Predict the output
                #print(f"Prediction: {y}, Target: {t}, Error (t - y): {t - y}")
                for i in range(len(self.weights)): # Loop through each of the weights indices
                    self.weights[i] += self.learning_rate * (t-y) * x[i]    # Apply the perceptron learning rule ;)
                #print(f"Updated weights: {self.weights}")

    """
    Evaluation method (evaluate):
        Input: (test_inputs, test_targets)
        For each input in the test set, use the predict method
        Compare the predicted output with the true target
        Compute the accuracy (the percentage of correct prediction)
    """

if __name__ == "__main__":
    # Initialize the Perceptron model with 9 input features and a learning rate of 1
    # Activation function: Bipolar = -1, Unipolar = 1

    # Set of <Tuple, Class> where L = 0 (or -1) and I = 1
    class_set = {}

    i_data = util.convert_txt_to_tuples("dataset_Is.txt")
    l_data = util.convert_txt_to_tuples("dataset_Ls.txt")

    for i in i_data:
        class_set[i] = 1

    # Ensure you match class_set[l] to the activation function you use
    for l in l_data:
        class_set[l] = -1

    noisey_i = util.convert_txt_to_tuples("noisey_Is.txt")
    noisey_l = util.convert_txt_to_tuples("noisey_Ls.txt")

    for i in noisey_i:
        class_set[i] = 1

    for j in noisey_l:
        class_set[j] = -1

    all_data = i_data + l_data + noisey_l + noisey_i
    # Ensure we dont have duplicates
    all_data = list(set(all_data))

    l5 = util.convert_txt_to_tuples("data5l.txt")
    i5 = util.convert_txt_to_tuples("data5i.txt")

    for i in i5:
        class_set[i] = 1

    for j in l5:
        class_set[j] = -1

    all_data = i5 + l5
    print(len(all_data))

    # shuffle them, but with same seed each time
    random.seed(9)
    random.shuffle(all_data)
    print(len(all_data), "total data-points")
    """
    max1 = 0

    for i in range(11, 300):
        p = Perceptron(input_size=26, learning_rate=1, a_function=-1)
        p.train(all_data, class_set, i)
        x = p.evaluate(all_data, class_set)
        if x == 100:
            print("IT CONVERGED")
            break

        if x > max1:
            max1 = x

    print("Max is", max1)
    """

    def partition_k_fold(dataset, k):
        length = len(dataset)
        f_size = length // k
        remainder = length % k

        folds = []
        start = 0

        for i in range(k):
            # If there is a remainder, distribute an extra element to some folds
            end = start + f_size + (1 if i < remainder else 0)
            folds.append(dataset[start:end])
            start = end

        return folds

    def cross_validation_k_fold(dataset, k, class_set, learning_rate=0.05, a_function=-1, epochs=50):
        folds = partition_k_fold(dataset, k)
        accuracies = []

        for i in range(k):
            # Use the all except 1 for training
            train_folds = [folds[j] for j in range(k) if j != i]
            test_fold = folds[i]

            train_vectors = []
            # Add each training tuple to the training list
            for fold in train_folds:
                train_vectors.extend(fold)

            # make a new perceptron, ensures we reset the weights
            p = Perceptron(input_size=len(train_vectors[0]), learning_rate=learning_rate, a_function=a_function)

            # Train the perceptron on the training set
            p.train(train_vectors, class_set, epochs=epochs)

            # Evaluate on the test fold
            accuracy = p.evaluate(test_fold, class_set)
            accuracies.append(accuracy)

            print(f"Accuracy for fold {i + 1}: {accuracy:.2f}%")

        # Compute and return the average accuracy
        avg_accuracy = np.mean(accuracies)
        #print(accuracies)
        print(f"\nAverage accuracy across {k} folds: {avg_accuracy:.2f}%")
        print("Min accuracy", min(accuracies))
        print("Max accuracy", max(accuracies))
        return avg_accuracy

    res = 0
    for i in range(30):
        res += cross_validation_k_fold(all_data, 12, class_set, .1, -1, 11)

    # see what the average of averages is
    print(res / 30)