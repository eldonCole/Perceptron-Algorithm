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
        
    def train(self, input_vectors, targets, epochs=100):
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
    In the following, we demonstrate the capability of this Perceptron class.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def choose_grid_size(): # This method prompts the user to choose between grid sizes, and how much variation of each character they would like to train and test the model on
    while True:                                                         # begin infinite loop until user inputs 3 or 5
        try:
            grid_size = int(input("Choose grid size (3 or 5): "))       # firstly, we prompt the user for either a 3x3, or a 5x5 matrix
            if grid_size in [3, 5]:                                     # if the input is a valid option, break from loop
                break
            else:
                print("Invalid choice. Please choose either 3 or 5.")   # prompt user to input valid DATA TYPE
        except ValueError:                                              # catch block to prompt for valid VALUE
            print("Invalid input. Please enter a number (3 or 5).")
    
    while True:                                                         # begin another loop until user inputs 1 or 2 for variations of characters
        try:
            version_choice = int(input("Would you like 1 version of each shape or 2 versions of each shape? (Enter 1 or 2): "))
            if version_choice in [1, 2]:                                # prompt the user, and break from loop onces 1 or 2 is entered
                break
            else:
                print("Invalid choice. Please choose either 1 or 2.")   # prompt user to input valid DATA TYPE
        except ValueError:                                              # catch block to prompt for valid VALUE
            print("Invalid input. Please enter a number (1 or 2).")
    
    return grid_size, version_choice                                    # return both user inputs (3 or 5, and 1 or 2)
# this method was the hardest part of the entire project.
def place_shape_in_grid(shape, grid_size):                              # This method places our defined variations of each character    
    rows, cols = shape.shape                                            # across the matrices. first, we invoke the .shape attribute of NumPys arrays that returns a tuple (3,3) or (5,5)
    positions = []                                                      # initialize an empty list for the various positions          
    for row in range(grid_size - rows + 1):                             # here we loop through the vertical positions of the grid, without exceeding the bounds
        for col in range(grid_size - cols + 1):                         # here, we loop through the horizontal posiotions of the grid, similarly
            grid = np.zeros((grid_size, grid_size), dtype=int)          # then we create an array of 0's with the size of the users choice, and declare int as data type
            grid[row:row + rows, col:col + cols] = shape                # specify the vertical slice beginning from row and going up to row+rows (exclusive). then do the same for columns
            positions.append(grid.flatten().tolist())                   # each time a grid is created, we flatten it to a list, and append it to our postions lists (list of lists)
    return positions                                                    # then return list of all positions

def main():                                            # main method, called to prompt user and initialize the perceptron operation
    grid_size, version_choice = choose_grid_size()     # call the choose grid size method to prompt the user

    L_shape = np.array([        # firstly, we must define our shapes (graphical representations of letters))
        [1, 0, 0],
        [1, 0, 0],              # here, we have a capital L, with bottom row size 3
        [1, 1, 1]
    ])
    alternate_L = np.array([    # here, we have an alternative L shape, with bottom row size 2 (cant really do lowercase because then it would be the same as our alternate i shape)
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])
    I_shape = np.array([        # then we have our capital I shape
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1]
    ])
    alternate_I = np.array([    # finally we have our alternate i shape.
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])

    L_positions = []    #initialize a list for all L positions
    I_positions = []    #initialize a list ofr all I positions

    if version_choice == 1:                                                 # if our user chose a single version of each character
        L_positions.extend(place_shape_in_grid(L_shape, grid_size))         # we invoke the place shape in grid method, and add the lists that it returns to the end of the positions lists
        I_positions.extend(place_shape_in_grid(I_shape, grid_size))         # then we do the same for our I shape
    elif version_choice == 2:                                               # else if the user chooses two versions of each shape
        L_positions.extend(place_shape_in_grid(L_shape, grid_size))         # we do the exace same except now we pass the place shape in grid method two versions of each shape
        L_positions.extend(place_shape_in_grid(alternate_L, grid_size))     # here, we pass alternate L
        I_positions.extend(place_shape_in_grid(I_shape, grid_size))
        I_positions.extend(place_shape_in_grid(alternate_I, grid_size))     # and here, we pass alternate I.

    all_vectors = L_positions + I_positions                                 # now that we have our populated lists of all positions, we place them into a single list of input vectors :)
    all_targets = [-1] * len(L_positions) + [1] * len(I_positions)          # then we populate a list of targets that MUST correlate with the list of input vectors, so we multiply by the
                                                                            # length of each lists (-1 for L's, 1 for I's)
                                                                            
    perceptron = Perceptron(input_size=grid_size * grid_size, learning_rate=1) # finally, we may initilize our perceptron, by specifying the input size, and learning rate
    perceptron.train(all_vectors, all_targets)                                 # next, we train the model with our extensive list of input vectors, and corresponding targets 
    perceptron.evaluate(all_vectors, all_targets)                              # last, but not least, we evaluate the intelligence of our model with the same inputs 

# invoke main function
if __name__ == "__main__":      # when a .py file is run, it has a built in __name__ variable, the default of which, is named "__main__". so this statemetn will always be true (unless we use
    main()                      # a module ran indirectly in another script). and thus, the main function is ran.
    
"""
    In this perceptron class, we find that it is very efficient in differentiation between our L shape, and I shape within a 3x3--even for our alternative shapes.
    However, when the model is tained on a 5x5 matrix, with TWO VERSIONS of each character, and the maximum number of positions for each character,
    it is unable to differentiate between our L's and I's.
    This tells us that--at least for these sets of alternate versions of characters--the data set of various positions and characters is unfortunately LINEARLY INSEPARABLE.
    This is at least the case for this given dataset, which is created by loop functions that generate all possible postions.
"""
