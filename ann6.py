import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Print the shape of the training data
print("Shape:", x_train.shape)

# Reshape the labels to one-hot vectors
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Define the class names
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Plot a sample image


# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create the ANN model
ann = models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(100, activation = 'relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10,activation='softmax'),
])

# Print a summary of the model
ann.summary()

# Compile the model
ann.compile(optimizer = 'adam',loss ='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
ann.fit(x_train, y_train, epochs=5)


def plot_sample(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])

plot_sample(x_train, y_train, 0)
