import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("data_2d.csv", header = None)

# Adding a column of all one's to dataframe
df[3] = 1.0

# Split the data into data & target
data = df[[0, 1, 3]].values
target = df[2].values

# w(xTx) = xTy
xT = np.transpose(data)

xTx = np.dot(xT, data)
xTy = np.dot(xT, target)

# np.linalg.solve(A, b). In our case A = xTx & B = xTy
w = np.linalg.solve(xTx, xTy)

# Now calculating the equation of line with minimum error!
Ypred = np.dot(data, w)

# Calculating R2. First calculating SSE & SST
SSE = np.sum(np.power(target - Ypred, 2))
SST = np.sum(np.power(target - np.mean(target), 2))

R2 = 1 - (SSE/SST)
print("The value of R-squared is:", R2)

# let's plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], target)
plt.show()