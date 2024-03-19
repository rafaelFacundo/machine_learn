import numpy as np
import matplotlib.pyplot as plt

#reading data from the dataset
data_from_artificial1d = np.genfromtxt(u"/home/rafaelfacundo/Documents/machine_learn/linear_regression/artificial1d.csv", delimiter=',')


column_x_from_data = data_from_artificial1d[:, 0]  # Coluna x, os valores de entrada
column_y_from_data = data_from_artificial1d[:, 1]  # Coluna y, os valores de saída para cada entrada de x

X = np.vstack((np.ones(len(column_x_from_data)), column_x_from_data)).T

# Calcular os coeficientes da regressão utilizando OLS
X_transpose = np.transpose(X)
W = np.linalg.inv(X_transpose @ X) @ X_transpose @ column_y_from_data

print("Coeficientes da regressão (interceptação e inclinação):", W)