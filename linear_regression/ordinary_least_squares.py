import numpy as np
import matplotlib.pyplot as plt

def normalization_minMax(xMatrice, numberOfColumns):
    X = xMatrice;
    print(X.shape)
    if numberOfColumns == 1:
        X = X.reshape((xMatrice.shape[0], 1))
    for col in range(numberOfColumns):
        maximum = X[:,col].max()
        minimum = X[:,col].min()
        X[:,col] += minimum
        X[:,col] *= (1/ (maximum - minimum))
    return X


def standartZeroNormalization(xMatrice, numberOfColumns):
    X = xMatrice
    if numberOfColumns == 1:
        X = X.reshape((xMatrice.shape[0], 1))
    for col in range(numberOfColumns):
        mean = X[:,col].mean()
        X[:,col] -= mean
        standarDeviation = np.std(X[:,col])
        X[:,col] *= (1/standarDeviation)
    return X

#reading data from the dataset
data_from_artificial1d = np.genfromtxt(u"/home/rafaelfacundo/Documents/machine_learn/lista_01_ama/artificial1d.csv", delimiter=',')
column_x_from_data = data_from_artificial1d[:, 0]  # Coluna x, os valores de entrada
column_y_from_data = data_from_artificial1d[:, 1]  # Coluna y, os valores de saída para cada entrada de x

column_x_from_data = normalization_minMax(column_x_from_data, 1);
column_y_from_data = standartZeroNormalization(column_y_from_data, 1);

X = np.hstack((np.ones((column_x_from_data.shape[0], 1)), column_x_from_data))

# Calcular os coeficientes da regressão utilizando OLS
X_transpose = np.transpose(X)
W = np.linalg.inv(X_transpose @ X) @ X_transpose @ column_y_from_data


print("Coeficientes da regressão:", W)