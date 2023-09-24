from multiple_linear_regression import mlr_regressor
from polynomial_regression import plr_regressor, poly_features
from support_vector_regression import svr_regressor, scalerX, scalerY
from decision_tree_regression import dtr_regressor
from random_forest_regression import rfr_regressor
from sklearn.metrics import r2_score
import pandas as pd
if __name__ == '__main__':
    # Importing the dataset
    test_dataset = pd.read_csv('Data_test.csv')
    X = test_dataset.iloc[:, :-1].values
    y = test_dataset.iloc[:, -1].values
    poly_X = poly_features.fit_transform(X)
    svr_X = scalerX.transform(X)

    # Predicting the values using all the different Regression Models!
    mlr_predict = mlr_regressor.predict(X)
    plr_predict = plr_regressor.predict(poly_X)
    svr_predict = svr_regressor.predict(svr_X)
    dtr_predict = dtr_regressor.predict(X)
    rfr_predict = rfr_regressor.predict(X)

    # Getting the r2_score of all the models
    print(f'Multiple Linear Regression Score -> {r2_score(y, mlr_predict)}')
    print(f'Polynomial Linear Regression Score -> {r2_score(y, plr_predict)}')
    print(f'Support Vector Regression Score -> {r2_score(scalerY.transform(y.reshape(-1, 1)), svr_predict)}')
    print(f'Decision Tree Regression Score -> {r2_score(y, dtr_predict)}')
    print(f'Random Forest Regression Score -> {r2_score(y, rfr_predict)}')
