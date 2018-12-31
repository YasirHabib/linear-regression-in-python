# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_excel("mlr02.xls")

# Adding a column of all one's to dataframe
df['X4'] = 1

# Split the data into data with 'X2' only as input
x2only = df[['X2', 'X4']].values
# Split the data into data with 'X3' only as input
x3only = df[['X3', 'X4']].values
# Split the data into data with both 'X2' & 'X3' as input
data = df[['X2', 'X3', 'X4']].values
# Split the data into target
target = df['X1'].values

def get_R2 (data, target):

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
	
get_R2 (x2only, target)
plt.scatter(x2only[:,0], target)
plt.title("Age vs Blood Pressure")
plt.xlabel("Age (years)")
plt.ylabel("Systolic blood pressure")
plt.show()

get_R2 (x3only, target)
plt.scatter(x3only[:,0], target)
plt.title("Weight vs Blood Pressure")
plt.xlabel("Weight (pounds)")
plt.ylabel("Systolic blood pressure")
plt.show()

get_R2 (data, target)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], target)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()