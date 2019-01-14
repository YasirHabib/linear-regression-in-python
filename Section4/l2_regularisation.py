# demonstration of L2 regularization

import numpy as np
import matplotlib.pyplot as plt

################################################################################################################
# X = np.array([1, 2])
# print(X.shape)				# (2,)
# print(X)						# [1 2]
# print(type(X))				# <class 'numpy.ndarray'>
# print(type(X[0]))				# <class 'numpy.int32'>
# print(len(X))					# 2
# print("\n")
################################################################################################################
# X = np.array([1, 2], [3, 4])	# TypeError: data type not understood
################################################################################################################
# X = np.array([[1, 2], [3, 4]])
# print(X.shape)				# (2, 2)
# print(X)						# [[1 2]
								#  [3 4]]
# print(type(X))				# <class 'numpy.ndarray'>
# print(type(X[0]))				# <class 'numpy.ndarray'>
# print(X[0])					# [1 2]
# print(type(X[0, 0]))			# <class 'numpy.int32'>
# print(X[0, 0])				# 1
# print(len(X))					# 2
# print("\n")
#################################################################################################################
# X = np.array([[1, 2, 3], [4, 5, 6]])
# print(X.shape)				# (2, 3)
# print(X)						# [[1 2 3]
								#  [4 5 6]]
# print(type(X))				# <class 'numpy.ndarray'>
# print(type(X[0]))				# <class 'numpy.ndarray'>
# print(X[0])					# [1 2 3]
# print(type(X[0, 0]))			# <class 'numpy.int32'>
# print(X[0, 0])				# 1
# print(len(X))					# 2
# print("\n")
#################################################################################################################
# a = [[1, 2, 3], [4, 5, 6]]
# print(a.shape)				# AttributeError: 'list' object has no attribute 'shape'
# print(a)						# [[1, 2, 3], [4, 5, 6]]
# print(type(a))				# <class 'list'>
# print(type(a[0]))				# <class 'list'>
# print(a[0])					# [1, 2, 3]
# print(type(a[0][0]))			# <class 'int'>
# print(a[0][0])				# 1
# print(len(a))					# 2
# print("\n")
##################################################################################################################

N = 50

# generate the data
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

# make outliers
Y[49] = Y[49] + 30              # also same as Y[-1] += 30
Y[48] = Y[48] + 30				# also same as Y[-2] += 30

# plot the data
plt.scatter(X, Y)
plt.show()

# add bias term
X = np.column_stack([X, np.ones(N)])		# We can alternatively also use X = np.vstack([X, np.ones(N)]).T

# plot the maximum likelihood solution
# w(xTx) = xTy
xT = np.transpose(X)
xTx = np.dot(xT, X)
xTy = np.dot(xT, Y)
w_ml = np.linalg.solve(xTx, xTy)			# This is the maximum likelihood solution
Ypred_ml = np.dot(X, w_ml)
plt.scatter(X[:,0], Y)
plt.plot(X[:,0], Ypred_ml)
plt.show()

# plot the map solution
# w(LI + xTx) = xTy							# L = Lambda
L = 1000.0
I = np.eye(2)								# We use identity matrix of 2x2 because it needs to have same dimensions as xTx which is also a 2x2 as can be
											# verified through the print(xTx) statement.
w_map = np.linalg.solve(L*I + xTx, xTy)
Ypred_map = np.dot(X, w_map)
plt.scatter(X[:,0], Y)
plt.plot(X[:,0], Ypred_ml, label="maximum likelihood")
plt.plot(X[:,0], Ypred_map, label = "map")
plt.legend()
plt.show()