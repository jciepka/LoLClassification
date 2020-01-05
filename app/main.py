import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from app.utils import plot_decision_regions
import matplotlib.pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

# load data to dataframe
df = pd.read_csv('./data/games.csv')

# # check data
# print(df.head())
# print(df.isnull().sum())
# # we have got some missing values so we want to delete rows with these missing values
# print(df.shape)
# # drop rows with missing values
df.dropna(inplace=True)
# print(df.shape)
# # we have deleted 4 rows


# split data to test and training sets
labels = df.loc[:, 'winner']
data = df.loc[:, df.columns != 'winner']
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=1)
data_combined = np.vstack((data_train, data_test))
labels_combined = np.hstack((labels_train, labels_test))

# FIRST MODEL - KNN
# for i in range(7):
#     knn = KNeighborsClassifier(n_neighbors=i + 1, p=2, metric='minkowski')
#     knn.fit(data_train, labels_train)
#     print('Accuracy of KNN: %.3f with number of neighbors: %d' % (knn.score(data_test, labels_test), i + 1))


# SECOND MODEL - NN WITH KERAS
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(60,)),
#     keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid),
# ])
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(data_train, labels_train, epochs=10, batch_size=5)
#
# test_loss, test_acc = model.evaluate(data_test, labels_test)
# print('Accuracy of NN:', test_acc)


# THIRD MODEL - XGBOOST

# model = XGBClassifier()
# model.fit(data_train, labels_train)
# # make predictions for test data
# y_pred = model.predict(data_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(labels_test, predictions)
# print("Accuracy of XGBoost: %.2f%%" % (accuracy * 100.0))
# print(model.max_depth)

# FOURTH MODEL - RANDOM FOREST

# rf = RandomForestRegressor(n_estimators=1000, random_state=42)
# rf.fit(data_train, labels_train)
# predictions = rf.predict(data_test)
# errors = abs(predictions - labels_test)
# mape = 100 * (errors / labels_test)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
#
# # Get numerical feature importances
# importances = list(rf.feature_importances_) # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(df.columns, importances)] # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) # Print out the feature and importances
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in importances];

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(data_train, labels_train)
y_pred = regressor.predict(data_test)

# errors = abs(y_pred - labels_test.values)
# mape = 100 * (errors / labels_test.values)
# accuracy = 100 - np.mean(mape)
print('Accuracy: %.3f' % regressor.score(data_test, labels_test))