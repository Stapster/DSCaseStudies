import pandas as pd
import numpy
import math
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Convolution1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
import dataprep.DataAnalysis2 as source

# https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/


# TODO - Konfigurationsanalyse aus dem folgenden Tutorial ausführen!
# https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/

# TODO - multivariate time series!!!
# https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57
# Anmerkung: möglichst alle (verschiedenen) Daten-Inputs getrennt normalisieren
# CNN ausprobieren

# TODO - interessant, stellt fest dass MLP für sequentielle Daten besser ist, aber CNN für Classification Porblems!!!
# https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a

oil_prices = source.OilData()
oil_prices.calculate_avg()
oil_prices.calculate_trend()
# oil_prices.normalize()

# univariater Datensatr
dataset = oil_prices.data["Avg"].values
dataset = dataset.reshape(dataset.shape[0], 1)

# print(dataset[0:5])
# Test ob die Trendberechnung funktioniert hat:
# print(oil_prices.data[["Trend", "Change"]].head())

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

################################################

# multivariate Daten
open_price = oil_prices.data["Open"].values.reshape(dataset.shape[0], 1)
close_price = oil_prices.data["Close"].values.reshape(dataset.shape[0], 1)
low_price = oil_prices.data["Low"].values.reshape(dataset.shape[0], 1)
high_price = oil_prices.data["High"].values.reshape(dataset.shape[0], 1)
volume = oil_prices.data["Volume"].values.reshape(dataset.shape[0], 1)
trend = oil_prices.data["Trend"].values.reshape(dataset.shape[0], 1)
change = oil_prices.data["Change"].values.reshape(dataset.shape[0], 1)

# Hier ggf. features entfernen/hinzufügen
dataset_multiv = [open_price, close_price, low_price, high_price, volume, change]
dataset_multiv_Y = trend
# print(dataset_multiv[0:5])
# for i in dataset_multiv:
#     print(i)

# train_size = int(len(close_price) * 0.67)
# test_size = len(close_price) - train_size
# trainX, testX = dataset_multiv[0:train_size], dataset_multiv[train_size:len(dataset_multiv)]
# trainY, testY = dataset_multiv_Y[0:train_size], dataset_multiv_Y[train_size:len(dataset_multiv_Y)]
# # trainX, testX = dataset_multiv[0:train_size, :], dataset_multiv[train_size:len(dataset_multiv), :]
# # trainY, testY = dataset_multiv_Y[0:train_size, :], dataset_multiv_Y[train_size:len(dataset_multiv_Y), :]
#
# print(len(trainX), len(trainY))
# print(len(testX), len(testY))
# print('----------------')
# print(trainX)


def create_dataset(dataset, look_back=1, forecast=1, sequence=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-(forecast-1)-(sequence-1)):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        b = dataset[(i + look_back + (forecast-1)):(i + look_back + (forecast-1)+(sequence-1)+1), 0]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)

# test_original = train[0:5]
# test_1X, test_1Y = create_dataset(test_original, 3, 1, 1)
# print(test_original)
# print('---- X-Data ----')
# print(test_1X)
# print('----- Y-Data ----')
# print(test_1Y)


def create_dataset_multivariate(dataset_multiv, dataset_multiv_Y, look_back=1, forecast=1, sequence=1):
    dataX, dataY = [], []

    for i in range(len(dataset_multiv[0]) - look_back - (forecast - 1) - (sequence - 1)):
        feature_values = []
        for feature_set in dataset_multiv:
            a = feature_set[i:(i + look_back), 0]
            feature_values.append(a)

        dataX.append(feature_values)
        b = dataset_multiv_Y[(i + look_back + (forecast - 1)):(i + look_back + (forecast - 1) + (sequence - 1) + 1), 0]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY), len(dataset_multiv)

test_6X, test_1Y, features = create_dataset_multivariate(dataset_multiv,dataset_multiv_Y, 5, 1, 1)

# print('---- Original-Data ----')
# for i in dataset_multiv:
#      print(i[0:10])
# print('---- X-Data ----')
# print(test_6X[0:6])
# print('----- Y-Data ----')
# print(test_1Y[0:6])
# print('----- Features ----')
# print(features)


def mlp_windowed():
    # reshape dataset
    look_back = 5
    forecast = 1
    sequence = 1
    trainX, trainY = create_dataset(train, look_back, forecast, sequence)
    testX, testY = create_dataset(test, look_back, forecast, sequence)
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


def mlp_windowed_trend():
    # reshape dataset
    look_back = 10
    forecast = 1
    sequence = 1
    trainX, trainY = create_dataset(train, look_back, forecast, sequence)
    testX, testY = create_dataset(test, look_back, forecast, sequence)

    model = Sequential()
    model.add(Dense(12, input_dim=look_back, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(sequence, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=400, batch_size=20, validation_data=(testX, testY), verbose=1)
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.show()

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: ', trainScore)
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: ', testScore)


def mlp_multivariate_trend():
    # reshape dataset
    look_back = 13
    forecast = 1
    sequence = 1
    trainX, trainY, features = create_dataset_multivariate(dataset_multiv, dataset_multiv_Y, look_back, forecast, sequence)

    # trainX, trainY = create_dataset(train, look_back, forecast, sequence)
    # testX, testY = create_dataset(test, look_back, forecast, sequence)

    print(trainX.shape)
    print(trainY.shape)
    print(trainX)
    print(trainY)

    model = Sequential()
    model.add(Dense(features+1, input_shape=(features, look_back), activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(sequence, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(trainX, trainY, validation_split=0.4, epochs=600, batch_size=12, verbose=2)
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.title('Multivariates Modell')
    pyplot.legend()
    pyplot.show()

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: ', trainScore)
    # testScore = model.evaluate(testX, testY, verbose=0)
    # print('Test Score: ', testScore)


# mlp_windowed()
# mlp_windowed_trend()
mlp_multivariate_trend()

########################
# Test für multivariate Datenformatierung

lookback = 1
forecast = 1
features = 4

openp = [10, 11, 12, 13, 12]
closep = [10.5, 11, 11.5, 12, 11.5]
volume = [30, 40, 35, 38, 45]
change = [0, 0, 1, 1, 0]

X, Y = [], []
for i in range(0, len(openp), 1):
    try:

        o = openp[i:i + lookback]
        c = closep[i:i + lookback]
        v = volume[i:i + lookback]
        x_i = change[i:i + lookback]

        y_i = change[i + forecast]

        x_i = numpy.column_stack((x_i, o, c, v))

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = numpy.array(X), numpy.array(Y)

#print(X)
#print(Y)

X_train = numpy.reshape(X, (X.shape[0], X.shape[1], features))

#print(X_train)
#print(Y)
