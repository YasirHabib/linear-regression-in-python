import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# Load the data
df = pd.read_csv("moore.csv", header = None, sep='\t')

#df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 102 entries, 0 to 101
# Data columns (total 6 columns):
# 0    102 non-null object
# 1    102 non-null object
# 2    102 non-null object
# 3    102 non-null object
# 4    97 non-null object
# 5    95 non-null object
# dtypes: object(6)
# memory usage: 4.9+ KB

#print(df.dtypes)
# 0    object
# 1    object
# 2    object
# 3    object
# 4    object
# 5    object
# dtype: object

#print(type(df[0]))
# <class 'pandas.core.series.Series'>

#print(type(df))
# <class 'pandas.core.frame.DataFrame'>

#print(df[[1, 2]])										# This returns columns with index 1 & 2

#print(df.columns)
# Int64Index([0, 1, 2, 3, 4, 5], dtype='int64')			# This gives us the datatype of the column index (not the contents which are above shown as object)

#print(df.columns.values)
# [0 1 2 3 4 5]

#print (type(df.iloc[0, 2]))							# Even though df.types returns object, the individual element of row 0, column 2 is a 'str'. This
# <class 'str'>                                           command is essential for determining the data type of individual elements.
#print (type(df.iloc[0, 1]))							# Even though df.types returns object, the individual element of row 0, column 1 is a 'str'. This
# <class 'str'>											  command is essential for determining the data type of individual elements.

df[2] = df[2].replace('\[.*\]', '', regex = True).astype(np.int64)     # \[.*\] is for removing references in square brackets
																	   # .astype(np.int64) is for converting from 'str' to 'int64'

df[1] = df[1].str.lstrip('~cca').str.rstrip('')						   # strips ~ & cca characters from data
df[1] = df[1].replace('\,', '', regex = True)						   # strips , from data
df[1] = df[1].replace('\[.*\]', '', regex = True).astype(np.int64)

#print (type(df.iloc[0, 2]))							# The individual element of row 0, column 2 is now a 'numpy.int64'
# <class 'numpy.int64'>
#print (type(df.iloc[0, 1]))							# The individual element of row 0, column 1 is now a 'numpy.int64'
# <class 'numpy.int64'>

# Split the data into data & target. We ignore column 0, 3, 4 & 5. Column 1 (no of transistors) is our target & column 2 (the year) is our data!
data = df[2].values
target = df[1].values

# To convert into log
target = np.log(target)

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

# how long does it take to double?
# log(transistorcount) = a*year + b
# transistorcount = exp(b) * exp(a*year)
# 2*transistorcount = 2 * exp(b) * exp(a*year) = exp(ln(2)) * exp(b) * exp(a * year) = exp(b) * exp(a * year + ln(2))
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
print("time to double:", np.log(2)/a, "years")

# Below is my solution for the above
# ln(Ypred) = a*data + b = a*year + b
# Ypred = exp(a*year + b) --> (I)
# 2Ypred = exp(a*yearWhenDoubled + b) --> (II)
# Now putting (I) into (II)
# 2 * exp(a*year + b) = exp(a*yearWhenDoubled + b)
# exp(ln(2) + a*year + b) = exp(a*yearWhenDoubled + b)
# Now if we take ln on both sides, the power comes down to the left & ln(exp) = 1
# ln(2) + a*year + b = a*yearWhenDoubled + b
# ln(2) = a*yearWhenDoubled - a*year
# ln(2) / a = yearWhenDoubled - year