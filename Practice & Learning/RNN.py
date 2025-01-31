# Name: Michael Hess
# Description

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData():
	def __init__(self, num_points, xmin, xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (xmax-xmin)/num_points
		self.x_data = np.linspace(xmin, xmax, num_points)
		self.y_true = np.sin(self.x_data)
	def ret_true(self, x_series):
		return np.sin(x_series)
	def next_batch(self, batch_size, steps, return_batch_ts=False):
		# Random starting point
		rand_start = np.random.rand(batch_size, 1)
		#Convert to be on time series
		ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))
		# Create batch time series on x axis
		batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution
		# Create y data for time series
		y_batch = np.sin(batch_ts)
		# Formatting for RNN
		if return_batch_ts:
			return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:,1:].reshape(-1, steps, 1), batch_ts
		else:
			return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:,1:].reshape(-1, steps, 1)

ts_data = TimeSeriesData(250, 0, 10)
# plt.plot(ts_data.x_data, ts_data.y_true)
# plt.show()
num_time_steps = 30
y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)

train_inst = np.linspace(5, 5 + ts_data.resolution*(num_time_steps+1), num_time_steps+1)

## CREATING THE MODEL ##

tf.reset_default_graph()
num_inputs = 1
num_neurons = 100
num_outputs = 1
lr = 0.0001
num_train_iterations = 2000
batch_size = 1

# Placeholders
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
# RNN Cell Layer
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_train_iterations):
		X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
		sess.run(train, feed_dict= {X: X_batch, y:y_batch})
		if i % 100 == 0:
			mse = loss.eval(feed_dict= {X:X_batch, y:y_batch})
			print(i, '\tMSE', mse)
	saver.save(sess, "./rnn_time_series_model")

with tf.Session() as sess:
	saver.restore(sess, "./rnn_time_series_model")
	X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
	y_pred = sess.run(outputs, feed_dict={X:X_new})


plt.title("Testing Model")
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), 'ko', label="Training Instance")
plt.plot(train_inst[1:], np.sin(train_inst[1:]), 'ro', label="Target")
plt.plot(train_inst[1:], y_pred[0,:,0], 'g')
plt.show()

## Generating New Sequence ##
with tf.Session() as sess:
	saver.restore(sess, "./rnn_time_series_model")
	# SEED ZEROS
	zero_seq_seed = [ 0.0 for i in range(num_time_steps)]
	for iteration in range(len(ts_data.x_data)-num_time_steps):
		X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, feed_dict={X:X_batch})
		zero_seq_seed.append(y_pred[0, -1, 0])
plt.plot(ts_data.x_data, zero_seq_seed, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], 'r')
plt.show()

with tf.Session() as sess:
	saver.restore(sess, "./rnn_time_series_model")
	# SEED ZEROS
	training_instance = list(ts_data.y_true[:30])
	for iteration in range(len(training_instance)-num_time_steps):
		X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, feed_dict={X:X_batch})
		training_instance.append(y_pred[0, -1, 0])
plt.plot(ts_data.x_data, ts_data.y_true, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], training_instance[:num_time_steps], 'r')
plt.show()