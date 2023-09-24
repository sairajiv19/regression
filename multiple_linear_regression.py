# Multiple Linear Regression Model

import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Splitting the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Training the Regression Model
mlr_regressor = LinearRegression()
mlr_regressor.fit(X, y)
