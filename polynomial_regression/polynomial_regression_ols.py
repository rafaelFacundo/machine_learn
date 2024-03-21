import numpy as np

data_from_artificial1d = np.genfromtxt(u"/home/rafaelfacundo/Documents/machine_learn/lista_01_ama/california.csv", delimiter=',')
inputValues = data_from_artificial1d[:, 0:7];
Yvalues = data_from_artificial1d[:, 8]
print(Yvalues)
#adding a column of ones in the input tables
#this is the artficial ones to match with the numbers of parameters
#this one will be on the side of W0
column_of_ones = np.ones((inputValues.shape[0], 1));
dataTable = np.hstack((column_of_ones, inputValues));
# now I gonna take 80% of the data to make the train set
# I will choice each data in an aleatory way
lengthOfTheTrainSet = int(0.8 * dataTable.shape[0]);
#generating aleatory indices to the train set
trainIndices = np.random.choice(dataTable.shape[0], lengthOfTheTrainSet, replace=False);
#Now getting the train set
trainSet = dataTable[trainIndices];
X = trainSet;
Xtranspose = X.T;


