# We use L1 regularisation if the dimensionality is greater or equal than the number of samples
# In this case our input 'X' is fat
# L1 regularisation encourages feature weights to be sparse i.e. only a few will be non-zero.
# L2 regularisation encourages feature weights to be small but will not encourage feature weights to go exactly zero.
# Both L1 & L2 regularisation help improve generalisation & test error because they prevent you from overfitting to the noise in the data. L1 does this
# by choosing the most important features, the ones that have the most impact with respect to the output. L2 does this by making the assertion that none
# of the weights are disproportionately large.

import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

#X = np.random.random((N, D))
#X = np.random.random((N, D)) - 0.5				# continuous uniform distribution centered around 0 from -0.5 to +0.5
X = (np.random.random((N, D)) - 0.5) * 10		# continuous uniform distribution centered around 0 from -5 to +5

# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

# generate Y - add noise with variance 0.5
Y = np.dot(X, true_w) + (np.sqrt(0.5) * np.random.randn(N))

# perform gradient descent to find w
costs = [] # keep track of squared error cost
w = np.sqrt(1/D) * np.random.randn(D) # randomly initialize w
learning_rate = 0.001
l1 = 10.0 # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?

for t in range(500):
	Ypred = np.dot(X, w)
	xT = np.transpose(X)
	w = w - learning_rate * (np.dot(xT, Ypred - Y) + l1 * np.sign(w))		# where dj/dw = np.dot(xT, Ypred - Y) + l1 * np.sign(w)
	
	# find and store the cost
	SSE = np.sum(np.power(Y - Ypred, 2))
	MSE = SSE / len(Y)
	
	costs.append(MSE)
	
# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot our w vs true w
plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()