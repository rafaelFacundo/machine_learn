import numpy as np

data_from_california = np.genfromtxt(u"/home/rafaelfacundo/Documents/machine_learn/lista_01_ama/california.csv", delimiter=',')

def raise_columns(columnXi, degree):
    #adding a column of 1 (ones) 
    newColumns = columnXi.reshape((columnXi.shape[0], 1))
    for p in range(2, degree + 1):
        columnXiRaisedToPowerP = np.power(columnXi,p);
        newColumns = np.hstack((newColumns, columnXiRaisedToPowerP.reshape((columnXi.shape[0], 1))));
    return newColumns

def create_x_matrice(dataMatrice, degree):
    X = np.ones((dataMatrice.shape[0], 1));
    numberOfColumns = dataMatrice.shape[1];
    for col in range(numberOfColumns):
        newColumnsRaisedToPowerP = raise_columns(dataMatrice[:,col], degree)
        X = np.hstack((X, newColumnsRaisedToPowerP));
    return X

def normalization_minMax(xMatrice):
    X = np.ones((xMatrice.shape[0], 1));
    for col in range(xMatrice.shape[1]):
        column = xMatrice[:,col];
        maximum = column.max()
        minimum = column.min()
        column = column + minimum
        column = column * (1/ (maximum - minimum))
        X = np.hstack((X, column.reshape((column.shape[0], 1))))
    return X


# now I gonna take 80% of the data to make the train set
# I will choice each data in an aleatory way
testSetSize = int(0.2 * data_from_california.shape[0])  
testIndices = np.random.choice(data_from_california.shape[0], testSetSize, replace=False)
conjunto_teste = data_from_california[testIndices]
trainSet = np.delete(data_from_california, testIndices, axis=0)

inputValues = trainSet[:, 0:8];
Yvalues = trainSet[:,8];

inputNormalized = normalization_minMax(inputValues)

print(inputValues);
print("+++++++++")
print(inputNormalized);
print("+++++++++")

X = create_x_matrice(inputNormalized, 3)
print(X)

X_transpose = np.transpose(X)
X_transposeTimesX = X_transpose @ X;
indentityMatrice = np.identity(X_transposeTimesX.shape[0]) * 0.0001;
X_transposeTimesX = np.add(X_transposeTimesX,indentityMatrice);
W = np.linalg.inv(X_transposeTimesX) @ X_transpose @ Yvalues; 

print("vector w: ")
print(W)