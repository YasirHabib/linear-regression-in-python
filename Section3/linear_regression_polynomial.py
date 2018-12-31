# shows how linear regression analysis can be applied to polynomial data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("data_poly.csv", header = None)

# Adding a column of all one's to dataframe
df[2] = 1.0

# Adding a column which is square of column[0]
df[3] = df.apply(lambda row: row[0] * row[0], axis=1)

# Split the data into data & target
data = df[[0, 3, 2]].values
target = df[1].values

# let's plot the data to see what it looks like
plt.scatter(data[:,0], target)						# I cannot use 'plt.scatter(data[0], target)' coz 'data' here is a numpy array & not a dataframe
plt.title("The data we're trying to fit")
plt.show()

# w(xTx) = xTy
xT = np.transpose(data)

xTx = np.dot(xT, data)
xTy = np.dot(xT, target)

# np.linalg.solve(A, b). In our case A = xTx & B = xTy
w = np.linalg.solve(xTx, xTy)

# Now calculating the equation of line with minimum error!
Ypred = np.dot(data, w)

# let's plot everything together to make sure it worked
plt.scatter(data[:,0], target)
plt.plot(data[:,0], Ypred, 'red')
plt.show()

# We use the sorted function to correct the points due to the montonically increasing quadratic function
plt.scatter(data[:,0], target)
plt.plot(sorted(data[:,0]), sorted(Ypred), 'red')
plt.show()

# Calculating R2. First calculating SSE & SST
SSE = np.sum(np.power(target - Ypred, 2))
SST = np.sum(np.power(target - np.mean(target), 2))

R2 = 1 - (SSE/SST)
print("The value of R-squared is:", R2)