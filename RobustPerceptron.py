import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:29:33 2024

@author(s): Cole Ralph, Austin Snyder, Bobi Vladimirov 
        Central Washington University
"""
class Perceptron:

    def __init__(self, input_size, learning_rate=1):            #Input size (number of features in the input), learning rate
        self.weights = np.random.rand(input_size).tolist()      #Initialize weights randomly (0,1)
        self.bias_weight = np.random.rand()                     #initialize random biases to increase model flexibility             
        self.learning_rate = learning_rate                      #set learning rate
        print("\nInitialized weights:", self.weights)           #display initial weights to user

    def predict(self, input_vector):
        x = input_vector                                        # Alias x for vector
        dot_product = 0                                         # Initialze dot_product
        print(f"\nPredicting for input vector: {x}")            # update user on prediction status
        for i in range(len(x)):                                 # Loop through indices
            dot_product = sum(x*w for x, w in zip(input_vector, # here, the zip functions takers multiple iterables and combines them into pairs of elements
                            self.weights)) + self.bias_weight   # then, we simply sum the products of the pairs and apply biases
            
        print(f"Dot Product: {dot_product}")                    # Display dot product before prediction
        if dot_product >= 0:                                    # If DP meets threshold, return 1,
            print("Prediction: 1 (dot product >= 0)\n")         
            return 1
        else:
            print("Prediction: -1 (dot product < 0)\n")         # else, return -1
            return -1
        
    def train(self, input_vectors, targets, epochs=30):
        for epoch in range(epochs):                                             # repeat for as many epochs as we choose
            print(f"\nEpoch {epoch + 1}")                                       # display epoch iteration to user
            for x, t in zip(input_vectors,targets):                             # Loop through the input and targets as tuples, so the model differentiates them
                print(f"\nTraining on input vector: {x}, target: {t}")          # display vector and target to user
                y = self.predict(x)                                             # Predict the output
                print(f"Prediction: {y}, Target: {t}, Error (t - y): {t - y}")  # display prediction to user
                error = t - y                                                   # calcuolate error
                self.weights = [w + self.learning_rate * error *                # apply the perceptron learning rule ;)
                                xi for w, xi in zip(self.weights, x)]           # [w = w + (learning_rate * error * xi)] where xi is the input values for the corresponding weight
                self.bias_weight += self.learning_rate * error                  # bias weight is adjusted separately. austin is a genius for including biases.

    def evaluate(self, input_vectors, targets):
        correct_predictions = 0                                     # Initialize correct inputs to 0
        
        for i, x in enumerate(input_vectors):                       # Iterate through the indecese with enumerate, much like zip, but with single iterable               
            prediction = self.predict(x)                            # make prediction
            
            if len(x) == 9:                                         # If input size is 9 (3x3 matrix)
                matrix = np.array(x).reshape(3, 3)                  # If input size is 25 (5x5 matrix)
            elif len(x) == 25:                                      # much more robust perceptron now :)
                matrix = np.array(x).reshape(5, 5)
            else:
                print(f"Unknown input vector size: {len(x)}")
                continue
            
            if prediction == -1:                                    # output prediction of shape so that we know
                print(f"The matrix:\n{matrix}\nrepresents an 'L'")  # the algorithm is associating correctly
            else:
                print(f"The matrix:\n{matrix}\nrepresents an 'I'")
            
            if prediction == targets[i]:                            # check if prediction matches target
                print("Prediction is correct.\n")
                correct_predictions += 1                            # increment correct predictions to calculate accuracy    
            else:
                print("Prediction is incorrect.\n")                 # else stupid ahh algorithm
    
        accuracy = correct_predictions / len(input_vectors) * 100   # calculate accuracy
        print(f"Accuracy: {accuracy:.2f}%")                         # display accuracy to user
        
"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    In the following, we procedurally demonstrate the capability of this Perceptron class.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
p = Perceptron(input_size=9, learning_rate=1)   # We begin by initializing the Perceptron model with 9 inputs (for a 3x3) and a learning rate of 1
matrix_x1 = [                                   #matrix_x1 represents L shape
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 1]
]
matrix_x2 = [                                   #matrix_x2 represents I shape
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1]
]

test_targets = [-1, 1]                  # Define the expected targets for each input vector: -1 for L, 1 for I
                                        # -1 for L, 1 for I
print("\nTraining Input Matrix x1:\n")  # Display input matrices for clarity
for row in matrix_x1:                   # loop through rows
    print(row)

print("\nTraining Input Matrix x2:\n")
for row in matrix_x2:
    print(row)
    
x1 = [item for row in matrix_x1 for item in row]        # Flatten each matrix into a 1D vector as required by the perceptron
x2 = [item for row in matrix_x2 for item in row]
input_vectors = [x1, x2]                                # initialize vector inputs (list of lists)
print("\nFlattened Vector form:\nx1", x1, "\nx2", x2)   # Display the flattened vectors for clarity

#p.train(input_vectors, test_targets)                    # Train our perceptron model on the input vectors and targets :)
#p.evaluate(input_vectors, test_targets)                 #evaluate whether the model is properly distinguishing the two
"""
  
#5x5 Model implementation----------------------------------------------------------------------------------------------------------------------------------------------------------


L_positions = []            # Now that our model is trained with 100% accuracy, lets test what it does with a 5x5 matrix.
L_shape = np.array([        # We will organize the same L shape from the 3x3 matrix in every possible location within the 5x5
    [1, 0, 0],              # There are 9 Possible ways to organize our L shape
    [1, 0, 0],
    [1, 1, 1]
])

for row in range(3):                                # We loop through all valid starting positions (row,col) in teh 5x5 grid. We can place the L in rows 0, 1, or 2 
    for col in range(3):                            # Then, we can place the L in columns o 1, or 2
        grid = np.zeros((5,5), dtype=int)           # Now, create a new 5x5 matrix of zeros
        grid[row:row+3, col:col+3] = L_shape        # Given that we are looping within range 3, "row:row+3" means rows 0, 1, or 2 ;)
        flattened_grid = grid.flatten().tolist()    #flatten our matrix to a vector (as list)
        L_positions.append(flattened_grid)          #append this grid to the list of possible L placements

L_targets = [-1] * len(L_positions)                 # We have now generated ever possilbe L shape position within a 5x5 matrix.
#p2 = Perceptron(input_size=25, learning_rate=1)    # Let's see how our model behaves when we train it with these inputs.
#p2.train(L_positions, L_targets)                   # This does also mean that we will have to alter our input size by initizlizing another perceptron
#p2.evaluate(L_positions, L_targets)                # Additoinally, we will have to extend the number of targets to match the number of L shape placements
                                                    # all targets should be -1 because it's all 'L'
                                                    
                                                    
I_positions = []            # Now that our 3x3 model is trained with 100% accuracy, lets test what it does with a 5x5 matrix.
I_shape = np.array([        # We will organize the same L shape from the 3x3 matrix in every possible location within the 5x5
    [1, 1, 1],              # There are 9 Possible ways to organize our L shape
    [0, 1, 0],
    [1, 1, 1]
])                                                    
for row in range(3):                                # We loop through all valid starting positions (row,col) in teh 5x5 grid. We can place the I in rows 0, 1, or 2 
   for col in range(3):                             # Then, we can place the I in columns o 1, or 2
        grid = np.zeros((5,5), dtype=int)           # Now, create a new 5x5 matrix of zeros
        grid[row:row+3, col:col+3] = I_shape        # Given that we are looping within range 3, "row:row+3" means rows 0, 1, or 2 ;)
        flattened_grid = grid.flatten().tolist()    # flatten our matrix to a vector (as list)
        I_positions.append(flattened_grid)          # append this grid to the list of possible I placements

I_targets = [1] * len(I_positions)                  # We have now generated ever possilbe I shape position within a 5x5 matrix.
#p2.train(I_positions, I_targets)                   # This does also mean that we will have to alter our input size by initizlizing another perceptron
#p2.evaluate(I_positions, I_targets)




LI_Targets = [-1] * len(L_positions) + [1] * len(I_positions)   # create array of targets for each possible position (necessary)
LI_Positions = L_positions + I_positions                        # create a list of all possible positions (list of vectors)

p3 = Perceptron(25,1)                                           # instantiate new perceptron with size 25 (5x5)
p3.train(LI_Positions,LI_Targets)                               # pass training parameters
p3.evaluate(LI_Positions,LI_Targets)                            # validate the model after it has been trained
            
            








