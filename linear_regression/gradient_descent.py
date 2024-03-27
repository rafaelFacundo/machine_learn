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

alpha  = 0.01 # alpha is the learning rate of our model
epochs = 1200 # number of iterations
w_zero = 0; # parameter of the model
w_one  = 0; # parameter of the model
column_x_from_data = data_from_artificial1d[:,0]; # taking column x, the column of the inputs
column_y_from_data = data_from_artificial1d[:,1]; # taking column y, the column of the output values for each input from x
N = column_x_from_data.shape[0];

column_x_from_data = standartZeroNormalization(column_x_from_data, 1);
column_y_from_data = standartZeroNormalization(column_y_from_data, 1);

print(column_x_from_data)

errorHistory = []

for t in range(epochs):    
    y_hat = w_zero + w_one * column_x_from_data;
    e_i   = column_y_from_data - y_hat;
    error = np.mean((e_i) ** 2)
    errorHistory.append(error)
    w_zero = w_zero + alpha * np.mean(e_i)
    w_one = w_one + alpha * np.mean(e_i * column_x_from_data)


""" plt.scatter(column_x_from_data[:, 0], column_y_from_data, color='blue', label='Dados de Treinamento')  
plt.plot(column_x_from_data[:, 0], y_hat, color='red', linewidth=2, label='Reta de Regressão') 
plt.xlabel('Variável Independente')  
plt.ylabel('Variável Dependente')
plt.title('Regressão Linear') 
plt.legend()  
plt.grid(True)  
plt.show()   """
plt.plot(range(1, epochs+1), errorHistory);
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Curva de Convergência da Regressão Linear')
plt.grid(True)
plt.show()

print("w0: ", w_zero);
print("W1: ", w_one)


