import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge, LSTM, Embedding
from keras.regularizers import l2, activity_l2
import keras
import numpy as np 
from sklearn import metrics
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def testNeurons():

	n = [2**i for i in range(7,8)]

	results = {}

	for i in range(len(n)):

		results[n[i]] = main(n[i])

	print(results)

def generateDataSep(file_name):

	data_left = []
	data_right = []

	labels = []

	i = 0
	with open(file_name, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ",", quotechar = "|")
		for row in reader:
			if i==0:
				i += 1
				# print(len(row[2:]))
				continue

			data_left.append(row[2:14] + row[27:])
			data_right.append(row[14:27])

			labels.append([row[1]])
	return np.asarray(data_left), np.asarray(data_right), np.asarray(labels)

def generateDataSet(file_name):

	data = []
	labels = []

	i = 0
	# input_ranges = [(2,6), (7,8), (12,16), (17,18), (22,24)]
	# with open(file_name, 'rb') as csvfile:
	# 	reader = csv.reader(csvfile, delimiter = ",", quotechar = "|")
	# 	for row in reader:
	# 		if i<430:
	# 			i += 1
	# 			# print(len(row[2:]))
	# 			continue
	# 		if i == 822:
	# 			break
	# 		current = []
	# 		for j in range(len(input_ranges)):

	# 			bounds = input_ranges[j]

	# 			for k in range(bounds[0], bounds[1]):

	# 				current.append(row[k])
	# 		data.append(current)
	# 		labels.append(int(row[1]))

	# 		i+= 1
	with open(file_name, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ",", quotechar = "|")
		for row in reader:
			if i<430:
				i += 1
				# print(len(row[2:]))
				continue
			if i == 822:
				break
			current = []
			for k in range(2, len(row)):

				current.append(float(row[k]))

			data.append(current)

			labels.append(int(row[1]))
			i += 1

	# print(data)
	# print(labels)

	return np.asarray(data), np.asarray(labels)

def generateDataLSTM(file_name, time_window):

	data = []

	labels = []

	i = 0
	with open(file_name, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ",", quotechar = "|")
		for row in reader:
			if i==0:
				i += 1
				# print(len(row[2:]))
				continue
			current = []
			for k in range(2, len(row)):

				current.append(float(row[k]))

			data.append(current)

			labels.append([int(row[1])])

	real_data = []

	for i in range(time_window, len(labels)):

		current = []

		for j in range(i-time_window, i):

			current.append(data[j])

		real_data.append(current)
 
	return np.asarray(real_data), np.asarray(labels[time_window:])

def logisticRegression():

	model = LogisticRegression()
	X, y = generateDataSet("normalizedRegression_removed.csv")
	# create the RFE model and select 3 attributes
	rfe = RFE(model, 12)
	rfe = rfe.fit(X, y)
	# summarize the selection of the attributes
	print(rfe.support_)
	print(rfe.ranking_)

	expected = y
	predicted = rfe.predict(X)
	# summarize the fit of the model
	print(metrics.classification_report(expected, predicted))

def LSTM_model():

	timesteps = 24
	data, labels = generateDataLSTM("normalizedKDA.csv", timesteps)

	model = Sequential()
	model.add(LSTM(32,
               input_shape=(timesteps, 22), dropout_W = 0.5))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(optimizer=keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0),
		              loss='binary_crossentropy',
		              metrics=['accuracy'])
	model.fit(data, labels , validation_split = 0.3, nb_epoch = 100, batch_size = len(data))

def main(n):
	
	data, labels = generateDataSet("normalizedKDA.csv")

	# dataTest = data[-75:]
	# labelsTest = labels[-75:]
	# data = data[:-75]
	# labels = labels[:-75]

	model = Sequential()
	# model.add(Dropout(0.4, input_shape = (22,)))
	model.add(Dense(n, input_dim = 22, activation = 'relu', W_regularizer =  keras.regularizers.WeightRegularizer(l1=0., l2=0.)))
	model.add(Dropout(0.5))
	model.add(Dense(n, activation = 'relu', W_regularizer =  keras.regularizers.WeightRegularizer(l1=0., l2=0.)))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(optimizer='adadelta',
		              loss='binary_crossentropy',
		              metrics=['accuracy'])
	history = model.fit(data, labels, validation_split = 0.2, nb_epoch = 1000, batch_size = len(data))
	# data_left, data_right, labels = generateDataSep("normalized.csv")

	# parameters = [0.001]

	# for p in parameters:
	# 	model = Sequential()
	# 	# model.add(Dropout(0.2, input_shape = (22,)))
	# 	model.add(Dense(n, input_dim = 22, activation = 'linear', W_regularizer = keras.regularizers.WeightRegularizer(l1=p, l2=0.)))
	# 	# model.add(Dense(n, activation='relu'))
	# 	# model.add(Dropout(0.2))
	# 	model.add(Dense(n, activation='linear', W_regularizer = keras.regularizers.WeightRegularizer(l1=p, l2=0.)))
	# 	# model.add(Dropout(0.2))
	# 	model.add(Dense(n, activation='linear', W_regularizer = keras.regularizers.WeightRegularizer(l1=p, l2=0.)))
	# 	# model.add(Dropout(0.2))
	# 	# model.add(Dropout(0.2))
	# 	model.add(Dense(1, activation = "sigmoid", W_regularizer = keras.regularizers.WeightRegularizer(l1=0, l2=0.)))
	# 	model.compile(optimizer='adadelta',
	# 	              loss='binary_crossentropy',
	# 	              metrics=['accuracy'])

	# 	model.fit(data, labels, validation_split=0.1, nb_epoch=250, batch_size=len(data))
		# print(model.layers[2].get_weights()[0])

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='lower right')
	plt.show()

	return None
	# return score
# LSTM_model()

# testNeurons()
main(128)
# logisticRegression()
