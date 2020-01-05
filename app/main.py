import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from app.utils import plot_decision_regions
import matplotlib.pyplot as plt

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
#     print('KNN Accuracy on test data: %.3f with number of neighbors: %d' % (knn.score(data_test, labels_test), i + 1))


# SECOND MODEL - NN WITH KERAS
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(60,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data_train, labels_train, epochs=5, batch_size=1)

test_loss, test_acc = model.evaluate(data_test, labels_test)
print('Test accuracy:', test_acc)
