import numpy as np

data_from_california = np.genfromtxt(u"/home/rafaelfacundo/Documents/machine_learn/lista_01_ama/california.csv", delimiter=',')
inputValues = data_from_california[:, 0:7];
Yvalues = data_from_california[:, 8]
""" print(Yvalues)
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
Xtranspose = X.T; """

""" 
    In this algorithm we will use OSL, but first we will make some modifications on the X matrice

"""

def createXiMatriceByPolinomialDegree(columnXi, degree):
    #adding a column of 1 (ones) 
    columnOfOnes = np.ones((columnXi.shape[0], 1));
    newXiMatrice = np.hstack((columnOfOnes, columnXi.reshape((columnXi.shape[0],1))));
    for p in range(2, degree + 1):
        columnXiRaisedToPowerP = columnXi ** p;
        newXiMatrice = np.hstack((newXiMatrice, columnXiRaisedToPowerP));
    return newXiMatrice

def createNewXMatrice(dataMatrice, degree):
    X_matrice = np.empty((0, dataMatrice.shape[0]))
    columnsOfDataMatrice = dataMatrice.shape[1]
    for column in range(columnsOfDataMatrice):
        print(column)
        Xi_column = dataMatrice[:,column];
        newMatriceFromXiColumn = createXiMatriceByPolinomialDegree(Xi_column, degree);
        X_matrice = np.vstack((X_matrice, newMatriceFromXiColumn.T))
       
   
    return X_matrice;

X = createNewXMatrice(inputValues, 1)

print(X)