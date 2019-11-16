# Name: Michael Hess
import pandas as pd
import numpy as np
import keras.utils
data = pd.read_csv('C:\\Users\\Micha\\Downloads\\letter-recognition.csv')
df = pd.DataFrame(data)

target = df["Letter"]
target = np.array(target)
target = [ord(letter) for letter in target]
target = keras.utils.to_categorical(target)

data = df.drop(["Letter"], axis=1)
data = np.array(data)


import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
model = models.Sequential()
num_inputs = 100
num_hidden = 150
num_outputs = 91
model.add(layers.Dense(units=num_inputs, activation=tf.nn.sigmoid, input_dim=16))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.sigmoid))
model.add(layers.Dense(units=num_outputs, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, target, epochs=100)

