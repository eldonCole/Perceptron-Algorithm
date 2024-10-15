def unipolar_activation(x):
    if x >= 1:
        print("Prediction: 1 (dot product >= 1 )\n")
        return 1
    print("Prediction: 0 (dot product < 1)\n")
    return 0

def bipolar_activation(x):
    if x >= 0:
        print("Prediction: 1 (dot product >= 0 )\n")
        return 1

    print("Prediction: -1 (dot product < 0)\n")
    return -1