import keras 
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN

nb_classes = 10

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = keras.utils.to_categorical(y_train, num_classes=nb_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes=nb_classes)

nb_units = 100

model = Sequential()

model.add(LSTM(nb_units, activation='relu', input_shape=(img_cols, img_rows)))
model.add(Dropout(0.25))

model.add(Dense(nb_units, activation='relu'))
model.add(Dropout(0.35))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 5

history = model.fit(X_train, 
                    Y_train, 
                    epochs=epochs, 
                    batch_size=64,
                    verbose=2)

scores = model.evaluate(X_test, Y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
