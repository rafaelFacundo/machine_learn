import numpy as np
import matplotlib.pyplot as plt

#reading data from the dataset
data_from_artificial1d = np.genfromtxt(u"/home/rafael/Documents/machine_learn/lista_01_ama/artificial1d.csv", delimiter=',')

alpha  = float(input("Type the learning rate: ")); # alpha is the learning rate of our model
epochs = int(input("Type the number of iterations(epochs): ")) # number of iterations
w_zero = 0; # parameter of the model
w_one  = 0; # parameter of the model
column_x_from_data = data_from_artificial1d[:,0]; # taking column x, the column of the inputs
column_y_from_data = data_from_artificial1d[:,1]; # taking column y, the column of the output values for each input from x
N = column_x_from_data.shape[0];
for t in range(epochs):    
    y_hat = w_zero + w_one * column_x_from_data;
    e_i   = column_y_from_data - y_hat;
    w_zero = w_zero + alpha * (np.sum(e_i) / N);
    e_i_times_column_x = e_i * column_x_from_data;
    w_one  = w_one + alpha * (np.sum(e_i_times_column_x) / N)


plt.scatter(column_x_from_data, column_y_from_data)
plt.plot(y_hat)

# Adicionando rótulos e título
plt.xlabel('Entradas (x)')
plt.ylabel('Saídas (y)')
plt.title('Gráfico de Pontos')

# Exibindo o gráfico
plt.show()


print("w0: ", w_zero);
print("W1: ", w_one)