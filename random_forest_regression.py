import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Splitting the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Random Forest Regression
rfr_regressor = RandomForestRegressor(n_estimators=10)
rfr_regressor.fit(X, y)
