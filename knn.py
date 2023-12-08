


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

# Load data 
project_data = pd.read_csv("credit_card.csv")

# separate data
X = project_data.drop('Class', axis =1)
y = project_data["Class"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# K-NN model
knn = KNeighborsRegressor(n_neighbors=12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# print performance
mse = metrics.mean_squared_error(y_test, y_pred)
print(mse)

