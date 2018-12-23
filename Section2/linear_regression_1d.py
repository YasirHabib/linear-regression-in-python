import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
X = pd.read_csv("data_1d.csv", header = None)

# Split the data into data & target
data = X[0].values
target = X[1].values

# This is to calculate the common denominator from variables 'a' & 'b' as solved in my notes!
d1 = np.dot(data, data) / len(data)
d2 = np.power(np.mean(data), 2)
denominator = d1 - d2

# Now calculating numerator for variable 'a' as solved in my notes!
num_a = (np.dot(data, target) / len(data)) - (np.mean(data) * np.mean(target))

# Now calculating numerator for variable 'b' as solved in my notes!
num_b = (np.mean(target) * np.dot(data, data) / len(data)) - np.mean(data) * (np.dot(data, target) / len(data))

# Now calculating variables 'a' & 'b' as solved in my notes!
a = num_a / denominator
b = num_b / denominator

# Now calculating the equation of line with minimum error!
Ypred = a*data + b

# Plotting the data on x-axis & target on y-axis
plt.scatter(data, target, marker = 'o', c = 'Black')
plt.title("Linear Regression")
plt.xlabel("Data")
plt.ylabel("Target")

# Plotting the line of best fit with minimum error
plt.plot(data, Ypred, 'red')
plt.show()

# Calculating R2. First calculating SSE & SST
SSE = np.sum(np.power(target - Ypred, 2))
SST = np.sum(np.power(target - np.mean(target), 2))

R2 = 1 - (SSE/SST)
print("The value of R-squared is:", R2)
