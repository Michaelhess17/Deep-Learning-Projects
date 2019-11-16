# Name: Michael Hess

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('C:\\Users\\Micha\\Downloads\\first-2\\fashion-mnist_train.csv')
y_true = np.array(data["label"])
x_data = np.array(data.drop(["label"], axis=1))


import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
model = models.Sequential()
num_inputs = 50
num_hidden = 100
num_outputs = 10
model.add(layers.Dense(units=num_inputs, activation=tf.nn.relu, input_dim=784))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.relu))
model.add(layers.Dense(units=num_outputs, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_true, epochs=100)



test_data = np.array(pd.read_csv('C:\\Users\\Micha\\Downloads\\first-2\\fashion_test_X.csv'))
output = model.predict_classes(test_data)

db = pd.DataFrame(output)
db.to_csv('C:\\Users\\Micha\\Downloads\\first-2\\results2.csv')
