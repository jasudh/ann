


import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)# def tanh(x):
    # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x-np.max(x,axis=0,keepdims=True)) 
    return exp_values / np.sum(exp_values,axis=0,keepdims=True)


x=np.linspace(-5,5,100)


plt.subplot(2,2,1)
plt.plot(x,sigmoid(x),label='sigmoid')
plt.title("Sigmoid Activation Function")
plt.legend()
plt.show()


plt.subplot(2,2,1)
plt.plot(x,relu(x),label='relu')
plt.title("relu Activation Function")
plt.legend()
plt.show()


plt.subplot(2,2,1)
plt.plot(x,tanh(x),label='tanh')
plt.title("tanh Activation Function")
plt.legend()
plt.show()


plt.subplot(2,2,1)
plt.plot(x,softmax(x),label='softmax')
plt.title("softmax Activation Function")
plt.legend()
plt.show()


