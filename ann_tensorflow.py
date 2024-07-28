#%% Importing the libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Importing the dataset

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

#%% Encoding the columns

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#%% Tranforming the columns

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#%% Splitting the dataset


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Normalizing the data

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#%% Building the ANN using tensorflow

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # First Hidden Layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Output Layer

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)


#%% Making the predictions

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

#%% Evaluating the results
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
