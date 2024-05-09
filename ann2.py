import numpy as np

def mcculloach_pits(inputs, weights, threshold):
    activation = np.dot(inputs, weights)
    output = 1 if activation >= threshold else 0
    return output

def ANDNOT(x1, x2):
    inputs = [x1, x2]
    weights = [1, -1]
    threshold = 1
    return mcculloach_pits(inputs, weights, threshold)

input1 = int(input("Enter the first input: "))
input2 = int(input("Enter the second input: "))
print(ANDNOT(input1, input2))
