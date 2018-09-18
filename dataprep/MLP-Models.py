import pandas as pd
import numpy
import math
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import dataprep.DataAnalysis2 as source

# https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

oil_prices = source.OilData()
oil_prices.calculate_avg()
# oil_prices.normalize()
dataset = oil_prices.data["Avg"].values
dataset = dataset.reshape(dataset.shape[0], 1)

# Generierung [-1,1] - Werte für Change
# change = oil_prices.data["Change"]
# c_arr = []
# for p in change[0:5]:
#     if p >= 0:
#         print(p, ' // ', 1)
#         c_arr.append(1)
#     else:
#         print(p, ' // ', -1)
#         c_arr.append(-1)
#
# print(c_arr)

# fix random seed for reproducibility
numpy.random.seed(7)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset_old(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-(look_forward-1)):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back + (look_forward-1), 0])
    return numpy.array(dataX), numpy.array(dataY)


def create_dataset(dataset, look_back=1, look_forward=1, sequence=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-(look_forward-1)-(sequence-1)):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        b = dataset[(i + look_back + (look_forward-1)):(i + look_back + (look_forward-1)+(sequence-1)+1), 0]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)


def mlp_basic ():
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=look_back, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)
    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


def mlp_windowed():
    # reshape dataset
    look_back = 5
    look_forward = 1
    sequence = 1
    trainX, trainY = create_dataset(train, look_back, look_forward, sequence)
    testX, testY = create_dataset(test, look_back, look_forward, sequence)
    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(12, input_dim=look_back, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(sequence))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=200, batch_size=2, validation_data=(testX, testY), verbose=2)
    # Validation-Data attribut und history ikl. Plot ist manuell eingefügt
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset, label="original", color="grey")
    plt.plot(trainPredictPlot, label="train prediction", color="green")
    plt.plot(testPredictPlot, label="test prediction", color="red")
    plt.legend(loc='lower left')
    plt.show()

    # newData = numpy.array([[65.5], [66.8], [67.0], [64.6], [65.0], [65.8]])
    # newPrediction = model.predict(newData)
    # plt.plot(newData, label="original", color="grey")
    # plt.plot(newPrediction, label="prediction", color="green")
    # plt.legend(loc='lower left')
    # plt.show()


# mlp_basic
# mlp_windowed()

test_original = train[0:5]
test_1X, test_1Y = create_dataset(test_original, 3, 1, 1)

# print(test_original)
# print('---- X-Data ----')
# print(test_1X)
# print('----- Y-Data ----')
# print(test_1Y)

