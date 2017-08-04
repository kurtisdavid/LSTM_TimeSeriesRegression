# credit to: http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/ for the basis of creating the structure

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.regularizers import l2, activity_l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def main(dataset, n_iter):
	# fix random seed for reproducibility
	np.random.seed(7)

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	# split into train and test sets
	train_size = int(len(dataset) * .2857)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)


	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(1, look_back,1)))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# model.fit(trainX, trainY, nb_epoch=n_iter, batch_size=1, verbose=2)
	for i in range(50):
		model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=2, shuffle=False)
		model.reset_states()

	trainPredict = model.predict(trainX, batch_size = 1)
	model.reset_states()
	testPredict = model.predict(testX, batch_size = 1)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	# unshifted original plots
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[:len(trainPredict), :] = trainPredict

	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict) + (look_back) + 1:len(dataset) - (look_back) -1, :] = testPredict

	return scaler.inverse_transform(dataset), testPredictPlot

# tests epoch size for each function
def test_functions():

	x_values = np.linspace(-2*np.pi, 6*np.pi, 200)
	datasets = []

	datasets.append(np.sin(x_values))
	datasets.append(np.sin(x_values) + np.cos(x_values))

	noisy = []
	for i in range(len(x_values)):

		if random.random()<.05:
			noisy.append(np.sin(x_values[i])+np.random.normal())
		else:
			noisy.append(np.sin(x_values[i]))

	datasets.append(np.array(noisy))

	inc_sin = []
	for i in range(len(x_values)):

		inc_sin.append(np.multiply(abs(x_values[i]), np.sin(x_values[i])))

	datasets.append(np.array(inc_sin))

	x_sin = []

	for i in range(len(x_values)):

		x_sin.append(np.add(x_values[i], np.sin(x_values[i])))

	datasets.append(np.array(x_sin))


	names = ["sin(x)", "sin(x)+cos(x)", "sin(x) [with noise]", "abs(x)*sin(x)", "x+sin(x)"]

	test_iterations = [50]

	for i in range(2,3):
		print(names[i])

		for j in range(len(test_iterations)):

			real, testPredictPlot = main(datasets[i].reshape([-1,1]), test_iterations[j])

			if (j==0):
				plt.plot(x_values, real, label = names[i])

			plt.plot(x_values, testPredictPlot, label = str(test_iterations[j]) + " epochs")

		plt.title(names[i])
		plt.xlabel("x")
		plt.ylabel("y")
		plt.legend(loc="upper right")
		plt.show()
		plt.clf()

test_functions()

	
