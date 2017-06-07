import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

trainData = [[0.001,0.02,0.02],[0.92,0.75,1],[0.31,1,0],[0.83,1,1],[1,0.012,1],[1,1,0.342]]
trainLabel = [0,0,0,1,1,1]
regr = linear_model.LinearRegression()
testData = [[1,0.37,0.33],[1,1,1]]
testLabel = [0,1]
regr.fit(trainData,trainLabel)
predicted = regr.predict(testData)
print("Coef_ = ")
print(regr.coef_)
print("\n")
print(predicted)