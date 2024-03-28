import numpy as np
import matplotlib.pyplot as plt

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
epochs = 600 # number of iterations
w_zero = 0; # parameter of the model
w_one  = 0; # parameter of the model
column_x_from_data = data_from_artificial1d[:,0]; # taking column x, the column of the inputs
column_y_from_data = data_from_artificial1d[:,1]; # taking column y, the column of the output values for each input from x
column_x_from_data_normalized = standartZeroNormalization(column_x_from_data, 1);
column_y_from_data_normalized = standartZeroNormalization(column_y_from_data, 1);
N = column_x_from_data.shape[0];
error_history = []
for t in range(epochs):
    # Embaralhar os dados
    permuted_indices = np.random.permutation(N)
    column_x_from_data_normalized = column_x_from_data_normalized[permuted_indices]
    column_y_from_data_normalized = column_y_from_data_normalized[permuted_indices]
    total_error = 0
    for i in range(N):
        y_hat = w_zero + w_one * column_x_from_data_normalized[i]
        e_i = column_y_from_data_normalized[i] - y_hat
        w_zero = w_zero + alpha * e_i
        w_one = w_one + alpha * e_i * column_x_from_data_normalized[i]
        total_error += e_i ** 2
    error_history.append(total_error / N)

print("w0: ", w_zero);
print("W1: ", w_one)

""" plt.scatter(column_x_from_data_normalized, column_y_from_data_normalized, color='blue', label='Dados de Treinamento')  
plt.plot(column_x_from_data_normalized, w_zero + w_one * column_x_from_data_normalized, color='red', linewidth=2, label='Reta de Regressão') 
plt.xlabel('Variável Independente')  
plt.ylabel('Variável Dependente')
plt.title('Regressão Linear') 
plt.legend()  
plt.grid(True)  
plt.show() 

print("w0:", w_zero)
print("w1:", w_one) """


plt.plot(range(1, epochs + 1), error_history)
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Curva de Convergência da Regressão Linear')
plt.grid(True)
plt.show() 

print("w0:", w_zero)
print("w1:", w_one)