import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Splitting the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Decision Tree Regression
dtr_regressor = DecisionTreeRegressor(random_state=0)
dtr_regressor.fit(X, y)
