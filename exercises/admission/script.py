# http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt # Scipy optimization API, similar to Matlab's fminunc
import pdb

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def cost(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log( sigmoid(X * theta.T) ))
	second = np.multiply( (1-y), np.log(1- sigmoid(X*theta.T)) )
	return np.sum(first-second)/(len(X))

def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

#data = pd.read_csv("ex4x.dat", delim_whitespace=True, names=["test1","test2"])
#data["admitted"] = pd.read_csv("ex4y.dat", delim_whitespace=True, names=["admitted"])

data = pd.read_csv("ex2data1.txt", header=None, names=["test1","test2", "admitted"])
data.head()
positive = data[data["admitted"].isin([1])]
negative = data[data["admitted"].isin([0])]

fig, ax = plt.subplots(figsize =(12,8))
ax.scatter(positive["test1"], positive["test2"], s=50, c="b", marker="o", label="Admitted")
ax.scatter(negative["test1"], negative["test2"], s=50, c="r", marker="x", label="Not Admitted")
ax.legend()
ax.set_xlabel("Test 1 Score")
ax.set_ylabel("Test 2 Score")

# Sigmoid Function
"""
nums = np.arange(-10, 10, step=1)
fig2, ax2 = plt.subplots(figsize=(12,8))  
ax2.plot(nums, sigmoid(nums), 'r')  
"""
plt.show()

data.insert(0, "Ones", 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)


val = cost(theta, X, y)
print "Val: ", val

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  
val2 = cost(result[0], X, y)

print "Val2:", val2

theta_min = np.matrix(result[0])  
predictions = predict(theta_min, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print 'accuracy = {0}%'.format(accuracy) 
