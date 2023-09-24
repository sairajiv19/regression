from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Splitting the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Polynomial Linear Regression
poly_features = PolynomialFeatures(degree=4)
poly_X = poly_features.fit_transform(X)
plr_regressor = LinearRegression()
plr_regressor.fit(poly_X, y)
