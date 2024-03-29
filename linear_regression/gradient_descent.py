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


errorHistory = []

for t in range(epochs):    
    y_hat = w_zero + w_one * column_x_from_data;
    e_i   = column_y_from_data - y_hat;
    error = np.mean((e_i) ** 2)
    errorHistory.append(error)
    w_zero = w_zero + alpha * np.mean(e_i)
    w_one = w_one + alpha * np.mean(e_i * column_x_from_data)

fig, (ax1, ax2) = plt.subplots(2)

ax1.scatter(column_x_from_data[:, 0], column_y_from_data, color='blue', label='Dados de Treinamento')  
ax1.plot(column_x_from_data[:, 0], y_hat, color='red', linewidth=2, label='Reta de Regressão') 
ax1.set_xlabel('Variável Independente')  
ax1.set_ylabel('Variável Dependente')
ax1.set_title('Regressão Linear') 
ax1.legend()  
ax1.grid(True)  

ax2.plot(range(1, epochs+1), errorHistory);
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Erro Quadrático Médio (MSE)')
ax2.set_title('Curva de Convergência da Regressão Linear')
ax2.grid(True)

plt.tight_layout()

plt.show()


print("w0: ", w_zero);
print("W1: ", w_one)


