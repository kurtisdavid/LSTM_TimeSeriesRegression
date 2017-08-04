import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math, random

def main(time_window, f, alpha):

	n = time_window
	n_hidden = 4
	
	# shape = [length of training set, n_inputs/outputs]
	x_ = tf.placeholder(tf.float32, [None, n], name = "x-input") #training inputs
	y_ = tf.placeholder(tf.float32, [None, 1], name = "y-input") #training outputs

	# shape = [# neurons from previous layer, # neurons in next layer]
	Theta1 = tf.Variable(tf.random_uniform([n,n_hidden], -1, 1), name="Theta1") 
	Theta2 = tf.Variable(tf.random_uniform([n_hidden,1], -1, 1), name="Theta2") 

	Bias1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
	Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")


	A2 = tf.nn.dropout(hiddenOutput(x_, Theta1, Bias1),1)	# hidden output
	Hypothesis = finalOutput(A2, Theta2, Bias2)	# second output layer

	#L1
	#+ .1*tf.reduce_sum(tf.abs(Theta1)) + .1*tf.reduce_sum(tf.abs(Theta2))

	#L2
	#+.5*tf.reduce_sum((Theta1)**2) + .5*tf.reduce_sum((Theta2)**2)

	# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
 #  	reg_constant = 0.01  # Choose an appropriate one. 
	cost = tf.reduce_sum(tf.squared_difference(Hypothesis, y_))/2 

	# add regularization

	# r = .4
	# if (f==noisy_sin):

	# 	cost += r*tf.reduce_sum((Theta1)**2) + r*tf.reduce_sum((Theta2)**2)

	train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
	inputs = create_function([],[],n,np.linspace(-2*np.pi-(4*np.pi/100*n), 2*np.pi, 100),f)
	input_x, input_y = inputs["x"], inputs["y"]

	if f==noisy_sin:

		for num_samples in range(5):
			new_inputs = create_function([],[],n,np.linspace(-2*np.pi-(4*np.pi/100*n), 2*np.pi, 100),f)
			input_x = input_x + new_inputs["x"]
			input_y = input_y + new_inputs["y"]
	print(input_x)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

        # start training
	for i in range(1000):
		for j in range(5):
			sess.run(train_step, feed_dict={x_: input_x[99*j:99*(j+1)], y_: input_y[99*j:99*(j+1)]})
		if i % 100 == 0:
			print('Epoch ', i)
			print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: input_x, y_: input_y}))
			print('Theta1 ', sess.run(Theta1))
			print('Bias1 ', sess.run(Bias1))
			print('Theta2 ', sess.run(Theta2))
			print('Bias2 ', sess.run(Bias2))
			print('cost ', np.sqrt(sess.run(cost, feed_dict={x_: input_x, y_: input_y})))

        # create test set
	test_x = np.linspace(2*np.pi-(2*np.pi/100*n), 4*np.pi,50)
        test_inputs = create_function([],[],n,test_x,f)
        test_input_x = test_inputs["x"]
	real_y = []

	for i in range(len(test_inputs["y"])):

		real_y.append(test_inputs["y"][i][0])
	
        # run test set into model
	test_A2 = tf.nn.dropout(hiddenOutput(test_input_x, Theta1, Bias1),1)
	test_hypothesis = sess.run(finalOutput(test_A2, Theta2, Bias2))
	test_y = []

	for i in range(len(test_hypothesis)):

		test_y.append(test_hypothesis[i][0])

	test_x = test_x[n:]

	return test_x, test_y, real_y

# prepare data for time series regression
def create_function(input_x, input_y, n, x_values, f):

	inputs = {"x": input_x, "y": input_y}

	for i in range(len(x_values)):

		if (i>=n):

			current = []

			for j in range(i-n,i):

				current.append(f(x_values[j]))

			inputs["x"].append(current)
			inputs["y"].append([f(x_values[i])])
			
	return inputs

# relu function
def hiddenOutput(x_, Theta1, Bias1):

	return tf.nn.relu(tf.matmul(x_, Theta1) + Bias1)

# final linear output
def finalOutput(A2, Theta2, Bias2):

	return tf.matmul(A2, Theta2) + Bias2

# perturbed sin function
def noisy_sin(x):

	p = random.uniform(-.1,.1)

	perturbed = np.sin(x) + p*np.sin(x)

	return perturbed

# cos(x) + sin(x)
def cos_plus_sin(x):

	return np.cos(x) + np.sin(x)

# |x|*sin(x)
def abs_sin(x):

	return np.multiply(abs(x), np.sin(x))

# x + sin(x)
def x_plus_sin(x):

	return np.add(x, np.sin(x))

# tests different time windows for MLP
def testTimeFrames():

	functions = [np.sin, cos_plus_sin, noisy_sin, abs_sin, x_plus_sin]

	window_sizes = [2]

	names = ["sin(x)", "sin(x)+cos(x)", "sin(x) [with noise]", "abs(x)*sin(x)", "x+sin(x)"]

	learning_rate = [.001,.001,.001,.0001,.0001]

	for i in range(2,3):

		for j in range(len(window_sizes)):

			x_range, test_y, real_y = main(window_sizes[j], functions[i], learning_rate[i])

			if (j==0):
				plt.plot(x_range, real_y, 'bo', label= names[i])

			graph_x = []
			graph_y = []

			for k in range(len(x_range)):

				if random.random()<.5:
					graph_x.append(x_range[k])
					graph_y.append(test_y[k])
			plt.plot(graph_x, graph_y, label= "Time window = " + str(window_sizes[j]))
		
		plt.legend(loc="upper right")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(names[i])
		plt.show()
		plt.clf()


testTimeFrames()
