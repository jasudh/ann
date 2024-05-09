import numpy as np

class BAM:
    def __init__(self, pattern1, pattern2):
        self.weights = np.dot(pattern1.T, pattern2)  # Compute weight matrix
    
    def recall(self, input_pattern):
        output_pattern = np.dot(input_pattern, self.weights)  # Compute output pattern
        output_pattern[output_pattern >= 0] = 1  # Threshold output pattern to 1 for positive values
        output_pattern[output_pattern < 0] = -1  # Threshold output pattern to -1 for negative values
        return output_pattern  # Return output pattern

# Define two pairs of vectors (patterns)
pattern1 = np.array([[1, 1, 1, -1], [-1, -1, 1, 1]])  # Pattern 1
pattern2 = np.array([[1, -1], [-1, 1]])  # Pattern 2

# Initialize BAM with the two pairs of vectors
bam = BAM(pattern1, pattern2)

# Take user input for input patterns
input_pattern1 = input("Enter input pattern 1 (separated by spaces): ").split()
input_pattern2 = input("Enter input pattern 2 (separated by spaces): ").split()

# Convert input patterns to numpy arrays
input_pattern1 = np.array(list(map(int, input_pattern1)))
input_pattern2 = np.array(list(map(int, input_pattern2)))

# Recall output patterns for input patterns
output1 = bam.recall(input_pattern1)  # Recall output pattern for input pattern 1
output2 = bam.recall(input_pattern2)  # Recall output pattern for input pattern 2

# Print output patterns
print("Output pattern for input pattern 1:", output1)
print("Output pattern for input pattern 2:", output2)