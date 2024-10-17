def unipolar_activation(x):
    if x >= 1:
        #print("Prediction: 1 (dot product >= 1 )\n")
        return 1
    #print("Prediction: 0 (dot product < 1)\n")
    return 0

def bipolar_activation(x):
    if x >= 0:
        #print("Prediction: 1 (dot product >= 0 )\n")
        return 1

    #print("Prediction: -1 (dot product < 0)\n")
    return -1

"""
 This reads in a txt file where each lines is a binary string of the flattened matrix.
 
 INPUTS: file_path => this is the path to the txt dataset 
 
 OUTPUTS: tuples_list => this is a list of tuples (vectors)
"""
def convert_txt_to_tuples(file_path):
    tuples_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Convert each character in the binary string to an integer
            int_tuple = tuple(int(digit) for digit in line.strip()) + (-1,)

            tuples_list.append(int_tuple)

    return tuples_list
