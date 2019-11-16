# Class: CS5 Gold
# File:
# Name: Michael Hess
# Description

from sklearn.datasets import load_wine
wine_data = load_wine()
# print(wine_data['DESCR'])


feat_data = wine_data['data']
labels = wine_data['target']
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.fit_transform(X_test)

import tensorflow as tf
from tensorflow import estimator
feat_cols = [tf.feature_column.numeric_column('x', shape=[13])]
deep_model = estimator.DNNClassifier(hidden_units=[13,13,13], feature_columns=feat_cols, n_classes=3,
                                     optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))
input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train}, y=y_train, shuffle=True, batch_size=10, num_epochs=5)

input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test}, shuffle=False)

preds = list(deep_model.predict(input_fn=input_fn_eval))
predictions = [p['class_ids'][0] for p in preds]
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predictions))
