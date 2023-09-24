import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Splitting the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Support Vector Regression
scalerX = StandardScaler()
scalerY = StandardScaler()
X = scalerX.fit_transform(X)
y = scalerY.fit_transform(y.reshape(-1, 1))
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X, y.ravel())