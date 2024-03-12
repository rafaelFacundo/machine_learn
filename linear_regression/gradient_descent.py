import numpy as np
import matplotlib.pyplot as plt

#reading data from the dataset
data_from_artificial1d = np.genfromtxt("./artificial1d.csv", delimiter=',')

alpha  = float(input("Type the learning rate: ")); # alpha is the learning rate of our model
epochs = int(input("Type the number of iterations(epochs): ")) 
W_zero = 0;
W_one  = 0;
t      = 0;

