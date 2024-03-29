import numpy as np
import matplotlib.pyplot as plt

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

def normalization_minMax(xMatrice, numberOfColumns):
    X = xMatrice;
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
        standartDeviation = np.std(X[:,col])
        X[:,col] *= (1/standartDeviation)
    return X, mean, standartDeviation


testSetSize = int(0.2 * data_from_california.shape[0])
testIndices = np.random.choice(data_from_california.shape[0], testSetSize, replace=False)
testSet = data_from_california[testIndices]
trainSet = np.delete(data_from_california, testIndices, axis=0)

testInputs = testSet[:, 0:8];
YvaluesTest = testSet[:,8];

inputValues = trainSet[:, 0:8];
Yvalues = trainSet[:,8];

inputValues = normalization_minMax(inputValues, inputValues.shape[1])
testInputs = normalization_minMax(testInputs, testInputs.shape[1])
yValuesNormalized, mean, standartDeviation = standartZeroNormalization(Yvalues, 1)

errorHisotryWithoutL2 = []
errorHisotryWithL2 = []

errorTestWithoutL2  = []
errorTestWithL2  = []

for p in range(1, 14):
    X = create_x_matrice(inputValues, p)
    Xtest = create_x_matrice(testInputs, p)

    X_transpose = np.transpose(X)
    X_transposeTimesX = X_transpose @ X;
    indentityMatrice = np.identity(X_transposeTimesX.shape[0]) * 0.01;
    W = np.linalg.inv(X_transposeTimesX) @ X_transpose @ yValuesNormalized; 
    y_hat = X @ W
    e_i = yValuesNormalized - y_hat
    error = np.sqrt(np.mean((e_i) ** 2));
    errorHisotryWithoutL2.append(error);

    y_hat = Xtest @ W;
    y_hat *= standartDeviation
    y_hat += mean
    e_i = YvaluesTest - y_hat
    error = np.sqrt(np.mean((e_i) ** 2));
    errorTestWithoutL2.append(error)


    X_transposeTimesX = np.add(X_transposeTimesX, indentityMatrice);
    W = np.linalg.inv(X_transposeTimesX) @ X_transpose @ yValuesNormalized; 
    y_hat = X @ W
    e_i = yValuesNormalized - y_hat
    error = np.sqrt(np.mean((e_i) ** 2));
    errorHisotryWithL2.append(error)

    y_hat = Xtest @ W;
    y_hat *= standartDeviation
    y_hat += mean
    e_i = YvaluesTest - y_hat
    error = np.sqrt(np.mean((e_i) ** 2));
    errorTestWithL2.append(error)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

ax1.plot(range(1, 14), errorHisotryWithoutL2);
ax1.set_xlabel("grau do polinomio");
ax1.set_ylabel("RMSE");
ax1.set_title("RMSE com dados de treino sem regularização");
ax1.grid(True);

ax2.plot(range(1, 14), errorHisotryWithL2);
ax2.set_xlabel("grau do polinomio");
ax2.set_ylabel("RMSE");
ax2.set_title("RMSE com dados de treino com regularização");
ax2.grid(True);

ax3.plot(range(1, 14), errorTestWithoutL2);
ax3.set_xlabel("grau do polinomio");
ax3.set_ylabel("RMSE");
ax3.set_title("RMSE com dados de teste sem regularização");
ax3.grid(True);

ax4.plot(range(1, 14), errorTestWithL2);
ax4.set_xlabel("grau do polinomio");
ax4.set_ylabel("RMSE");
ax4.set_title("RMSE com dados de teste com regularização");
ax4.grid(True);

plt.tight_layout();

plt.show();