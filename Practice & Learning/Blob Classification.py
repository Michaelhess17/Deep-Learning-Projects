# Name: Michael Hess


from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100000, centers=50, n_features=200, random_state=0)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
model = models.Sequential()
num_inputs = 50
num_hidden = 100
num_outputs = 50
model.add(layers.Dense(units=num_inputs, activation=tf.nn.relu, input_dim=200))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden, activation=tf.nn.relu))
model.add(layers.Dense(units=num_outputs, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


loss, acc = model.evaluate(X_test, y_test)
print("Model Accuracy: {} \n Model Loss: {}".format(acc, loss))



