# Instead of taking derivative (gradient) of cost/error function & then setting it to 0 to find the value of the weight(w), we here initialise 'w' with 
# a random value, & then iteratively update 'w' in the direction of the derivative (gradient) of the cost/error (dj/dw) in small steps. Hence that's why
# the method is called gradient descent.

import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N, D))
X[:, 0] = 1												# replace zeros in column 0 with ones. This is the bias term.
X[:5, 1] = 1											# replace first 5 zeros in column 1 with ones
X[5:, 2] = 1											# replace last 5 zeros in column 2 with ones

Y = np.array([0]*5 + [1]*5)								# creates a vector of 5 zeros and 5 ones

# print X so you know what it looks like
print("X:", X)

# xT = np.transpose(X)
# xTx = np.dot(xT, X)
# xTy = np.dot(xT, Y)
# w = np.linalg.solve(xTx, xTy)							# The regular solution gives error because of singular matrix hence we use gradient descent


# let's try gradient descent
# Initiaise 'w' to some random value. A good initialisation is to draw a sample from a gaussian centered at 0 & variance 1/D.
# The number of samples have to be equal to D = Dimensionality.
w = np.sqrt(1/D) * np.random.randn(D)                   # The term to the left of random.randn is for standard deviation. So, to have a variance of 1/D,
														# we need a standard deviation of sqrt(1/D)
# print(w.shape)										# returns (3,). Remember weights (w) are always a vector with elements equal to D = Dimensionality.
learning_rate = 0.001
costs = [] 												# keep track of squared error cost

for t in range(1000):
	Ypred = np.dot(X, w)
	xT = np.transpose(X)
	w = w - learning_rate * (np.dot(xT, Ypred - Y))		# where dj/dw = np.dot(xT, Ypred - Y)
	
	SSE = np.sum(np.power(Y - Ypred, 2))
	MSE = SSE / len(Y)
	
	costs.append(MSE)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot prediction vs target
plt.plot(Ypred, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()