import numpy as np
import matplotlib.pyplot as plt

num_epochs = int(input("Enter the number of training epochs:"))
learning_rate = float(input("Enter the learning rate"))

x_train = np.array([[2,4],[4,6],[5,2],[6,2],[5,7],[7,3],[8,4]])
y_train = np.array([1,1,1,1,-1,-1,-1])

weights = np.random.rand(2)
bias = np.random.rand()

for _ in range(num_epochs):
    for inputs,label in zip(x_train,y_train):
        summation = np.dot(inputs,weights)+bias
        activation = 1 if summation>=0 else -1
        weights += learning_rate*(label-activation)*inputs
        bias += learning_rate * (label-activation)

plt.figure(figsize=(8,6))
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)

x = np.linspace(0,10,100)
if weights[1]!=0:
    y = -(weights[0]*x+bias)/weights[1]
    plt.plot(x,y,color='red',label="Decision boundary")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("perceptron decision region")
    plt.legend()
    plt.show()
        
