# Find best fit for a data in the form of a sine wave.
# Now, since sine wave is not linear, you will have to use a 'polynomial line' to find a best fit & not a 'straight line'.
# So everytime the data is not a straight line, you introduce a polynomial of a certain degree. The whole purpose of this exercise is to remember that
# that increasing degree (or power) beyond a certain number starts increasing the test error.

# Everything depends on the training data. If your training data is fully representative of all the rest of the data, then the fit will be good.
# In terms of our sine wave, if the training data is spread out so that we get an evenly spread out sample for all values of X, then our polynomial will
# fit well. That's because our training data is a good representative of our test data. However, if our training data is all bunched up in a few spots, then
# you see that the polynomial goes wild in spots where there wasn't any training data. That's because the training data was not a good representative of the 
# test data. So the moral of the story is to collect lots of data. The more data we have, the better it represents reality and the more accurate our models
# will be on unseen data.

import numpy as np
import matplotlib.pyplot as plt

# Return random floats in the half-open interval [0.0, 1.0). Results are from the “continuous uniform” distribution

#R = np.random.random((10, 10))
#print(R.mean())
# Should return ~ 0.5
#print(R.var())
# Should return ~ 0
#plt.hist(R, bins = 10)
#plt.show()

# Return a sample (or samples) from the “standard normal” distribution

#G = np.random.randn(10, 10)
#print(G.mean())
# Should return ~ 0
#print(G.var())
# Should return ~ 1
#plt.hist(G, bins = 10)
#plt.show()

#A = np.zeros((5, 3))
#print(A)
#B = np.append(A, np.ones((5, 1), dtype=np.int64), axis=1)
#print(B)

# To add polynomial terms to the original input to get polynomial regression. This function adds all the polynomials from degree 1 upto the degree 
# specified in the input arguments. It then adds a column of ones in the end.
def make_polynomial(data, degree):

	poly = []
	for x in range(degree):
		poly.append(data**(x+1))
	poly.append(np.ones(len(data)))
	poly = np.column_stack(poly)
	return poly

# fit function which is just the solution to linear regression for finding the weights in terms of the inputs & the outputs.	
def fit(data, target):
	# w(xTx) = xTy
	xT = np.transpose(data)

	xTx = np.dot(xT, data)
	xTy = np.dot(xT, target)

	# np.linalg.solve(A, b). In our case A = xTx & B = xTy
	w = np.linalg.solve(xTx, xTy)
	return w

# This function takes input X, Y, sample & degree. The sample argument will tell us the number of samples to take from X & Y to form a training set.
# We will then use that data to find the best weights for a polynomial of the degree specified in the argument. We then plot that polynomial by using
# 'plt.plot(X, Ypred)'. We also plot the original sine wave by using 'plt.plot(X, Y)' & the training samples by using 'plt.scatter(Xtrain, Ytrain)'.
# This should gives us an idea of both how well the polynomial fits to the training data & how well it generalises to the entire sine wave that it has
# not seen before.
# My comments: Would it be correct to say that we want to fit (calculating 'w') to training data & generalise (predicting) to test data??????
def fit_and_display(X, Y, sample, degree):  # X, Y, 10, 5
	train_idx = np.random.choice(len(X), sample)
	Xtrain = X[train_idx]
	Ytrain = Y[train_idx]
	
	plt.scatter(Xtrain, Ytrain)							# random training samples of 10 points
	plt.show()
	
	Xtrain_poly = make_polynomial(Xtrain, degree)
	w = fit(Xtrain_poly, Ytrain) 						# Only training data should be used during fitting.
	
	data_poly = make_polynomial(X, degree)
	Ypred = np.dot(data_poly, w)						# Non-training data (called validation or test data) is used for evaluation.
	
	plt.plot(X, Y)
	plt.plot(X, Ypred)
	plt.scatter(Xtrain, Ytrain)
	plt.title("deg = %d" % degree)
	plt.show()

# This function calculates mean square error which we are going to plot for both train & test sets at the same time.
# You will see that for a while the test error will fall with the train error but eventually the test error will blow up because the polynomial does
# not know what to do with the inputs that it hasn't seen before. So the next function (plot_train_vs_test_curves) does just that.
# It takes a random sample of X & Y, makes that the train set & treats all the rest as the test set. It then fits polynomials all the way from degree
# 1 upto degree 20 & plots the train & test error curves. Since the test error curve will be really large & might overtake the train error curve, we are
# also going to plot the train error curve by itself so you can see that it falls to zero & then it stays there.
def get_mse(Y, Ypred):
	SSE = np.sum(np.power(Y - Ypred, 2))
	MSE = SSE / len(Y)
	return MSE

def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    train_idx = np.random.choice(len(X), sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    test_idx = []	
    for idx in range(len(X)):
        if idx not in train_idx:
            test_idx.append(idx)           

    # test_idx = np.random.choice(N, sample)
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []
    for deg in range(max_deg+1):
        Xtrain_poly = make_polynomial(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Ypred_train = np.dot(Xtrain_poly, w)
        mse_train = get_mse(Ytrain, Ypred_train)

        Xtest_poly = make_polynomial(Xtest, deg)
        Ypred_test = np.dot(Xtest_poly, w)
        mse_test = get_mse(Ytest, Ypred_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)
		
    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.show()
	
# Create a sine wave	
X = np.linspace(0, 6*np.pi, 100)
Y = np.sin(X)
plt.plot (X, Y)
plt.show()

for degree in (5, 6, 7, 8, 9):
    fit_and_display(X, Y, 10, degree)
	
plot_train_vs_test_curves(X, Y)

