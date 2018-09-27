import numpy
import math
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import dataprep.DataAnalysis2 as source

################################################

# Daten einlesen und data-split durchführen

################################################

# Datensatz einlesen und initial modifizieren
oil_prices = source.OilData()
oil_prices.calculate_avg()
oil_prices.calculate_trend()
oil_prices.pvt()
oil_prices.rsi()
oil_prices.macd()
oil_prices.normalize()
# oil_prices.log_transform()

# univariater Datensatz
dataset = oil_prices.data["Avg"].values
dataset = dataset.reshape(dataset.shape[0], 1)

# print(dataset[0:5])
# Test ob die Trendberechnung funktioniert hat:
# print(oil_prices.data[["Trend", "Change"]].head())

# split in Trainings- und Testdaten (bzw. Validierungsdaten)
split = 0.67
train_size = int(len(dataset) * split)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Univariate data-split")
print("Train: ", len(train), " / Test: ", len(test))
print("--------------------------")

################################################

# multivariate Daten
open_price = oil_prices.data["Open"].values.reshape(dataset.shape[0], 1)
close_price = oil_prices.data["Close"].values.reshape(dataset.shape[0], 1)
low_price = oil_prices.data["Low"].values.reshape(dataset.shape[0], 1)
high_price = oil_prices.data["High"].values.reshape(dataset.shape[0], 1)
volume = oil_prices.data["Volume"].values.reshape(dataset.shape[0], 1)
trend = oil_prices.data["Trend"].values.reshape(dataset.shape[0], 1)
change = oil_prices.data["Change"].values.reshape(dataset.shape[0], 1)

# zusätzliche Indikatoren
oil_prices.data["rsi"] = oil_prices.data["rsi"].astype(float)
oil_prices.data["macd_val"] = oil_prices.data["macd_val"].astype(float)
oil_prices.data["pvt"] = oil_prices.data["pvt"].astype(float)
rsi = oil_prices.data["rsi"].values.reshape(dataset.shape[0], 1)
macd_val = oil_prices.data["macd_val"].values.reshape(dataset.shape[0], 1)
pvt = oil_prices.data["pvt"].values.reshape(dataset.shape[0], 1)

# rsi = numpy.log(rsi)
# macd_val = numpy.log(macd_val)
# pvt = numpy.log(pvt)

# Hier Features entfernen/hinzufügen
dataset_multiv = [open_price, close_price, low_price, high_price, volume, change]
dataset_multiv_Y = trend
dataset_multiv_Y_REG = close_price

# data-split
split = 0.8
train_size = int(len(dataset_multiv[0]) * split)
test_size = len(dataset_multiv[0]) - train_size

# Input-Variablen
trainX_multiv = []
testX_multiv = []
for i in dataset_multiv:
    trainX, testX = i[0:train_size], i[train_size:len(i)]
    trainX_multiv.append(trainX)
    testX_multiv.append(testX)

# Target-Vektoren
trainY_multiv, testY_multiv = dataset_multiv_Y[0:train_size], dataset_multiv_Y[train_size:len(dataset_multiv_Y)]
trainY_multiv_REG, testY_multiv_REG = dataset_multiv_Y_REG[0:train_size], dataset_multiv_Y_REG[train_size:len(dataset_multiv_Y_REG)]

print("Multivariate data-split to ", train_size, "/", test_size)
print("X-Values // Train: ", len(trainX_multiv[0]), " / Test: ", len(testX_multiv[0]))
print("Y-Trend  // Train: ", len(trainY_multiv), " / Test: ", len(testY_multiv))
print("Y-REG    // Train: ", len(trainY_multiv_REG), " / Test: ", len(testY_multiv_REG))
print("--------------------------")

################################################

# Data-frames für MLP herstellen

################################################

# Erstellt generisch Datensätze für das MLP mit den 3 wichtigsten Merkmalen:
# look_back: Anzahl der berücksichtigten Tage vor der Prediction
# forecast: Welcher Wert t+x soll predicted werden
# sequence: Anzahl der Werte ab t+x, die predicted werden sollen


# Einfacher dataframe mit nur 1 Feature
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


# Komplexer dataframe mit beliebig vielen Features
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

# test_6X, test_1Y, features = create_dataset_multivariate(dataset_multiv,dataset_multiv_Y, 5, 1, 1)
# print('---- Original-Data ----')
# for i in dataset_multiv:
#      print(i[0:10])
# print('---- X-Data ----')
# print(test_6X[0:6])
# print('----- Y-Data ----')
# print(test_1Y[0:6])
# print('----- Features ----')
# print(features)

################################################

# Modellierung

################################################


def mlp_univariate():
    # Modellparameter festlegen
    look_back = 5
    forecast = 1
    sequence = 1

    # Vektoren generieren
    trainX, trainY = create_dataset(train, look_back, forecast, sequence)
    testX, testY = create_dataset(test, look_back, forecast, sequence)

    # Modell konfigurieren und generieren
    model = Sequential()
    model.add(Dense(8, input_dim=look_back, activation='relu'))
    model.add(Dense(sequence))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Modell fitten und Entwicklung plotten
    history = model.fit(trainX, trainY, epochs=200, batch_size=8, validation_data=(testX, testY), verbose=0)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Modell evaluieren (Scores berechnen und ausgeben)
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


def mlp_univariate_trend():
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


def mlp_multivariate(look_back=13, forecast=1, sequence=1, numberEpochs=500, batch=12):
    # Trainings- und Testdaten generieren
    trainX, trainY, features = create_dataset_multivariate(trainX_multiv, trainY_multiv_REG, look_back, forecast, sequence)
    testX, testY, features = create_dataset_multivariate(testX_multiv, testY_multiv_REG, look_back, forecast, sequence)

    # print(trainX.shape)
    # print(trainY.shape)

    # Modell konfigurieren und generieren
    model = Sequential()
    model.add(Dense(features+1, input_shape=(features, look_back), activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(sequence))
    model.compile(loss='mean_squared_error', optimizer='adam')

    stop_criteria = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=numberEpochs/2, verbose=0, mode='auto')
    mcp = ModelCheckpoint("model_best.hdf5", monitor="val_loss",
                          save_best_only=True, save_weights_only=False, verbose=0)
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=numberEpochs, batch_size=batch,
                        verbose=0, callbacks=[mcp, stop_criteria])

    del model
    best_model = load_model("model_best.hdf5")

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.ylim(0, 30)
    pyplot.xlim(0, numberEpochs)
    pyplot.title('Multivariates Modell: Regression (MSE) / ' + str(look_back) + ' / ' + str(forecast) + ' / '
                 + str(sequence) + ' / ' + str(numberEpochs) + ' / ' + str(batch))
    pyplot.legend()
    pyplot.show()

    # Model evaluieren
    if oil_prices.log_transformed:
        prediction = best_model.predict(testX)
        prediction_bt = numpy.exp(prediction)
        testY_bt = numpy.exp(testY)
        mse = ((prediction_bt - testY_bt) ** 2).mean()
        print('Test Score: ', mse)
    else:
        trainScore = best_model.evaluate(trainX, trainY, verbose=0)
        print('Train Score: ', trainScore)
        testScore = best_model.evaluate(testX, testY, verbose=0)
        print('Test Score: ', testScore)

    # Test
    # pyplot.plot(prediction_bt, label='prediction')
    # pyplot.plot(testY_bt, label='original')
    # pyplot.title('Multivariates Modell: Prediction vs. Original / ' + str(look_back) + ' / ' + str(forecast) + ' / '
    #              + str(sequence) + ' / ' + str(numberEpochs) + ' / ' + str(batch))
    # pyplot.legend()
    # pyplot.show()

    return best_model


def mlp_multivariate_trend(look_back=13, forecast=1, sequence=1, numberEpochs=500, batch=12):
    # Trainings- und Testdaten generieren
    trainX, trainY, features = create_dataset_multivariate(trainX_multiv, trainY_multiv, look_back, forecast, sequence)
    testX, testY, features = create_dataset_multivariate(testX_multiv, testY_multiv, look_back, forecast, sequence)

    # print(trainX.shape)
    # print(trainY.shape)

    # Modell konfigurieren und generieren
    model = Sequential()
    model.add(Dense(features+1, input_shape=(features, look_back), activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(sequence, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    stop_criteria = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=numberEpochs/2, verbose=0, mode='auto')
    mcp = ModelCheckpoint("model_best.hdf5", monitor="val_acc",
                          save_best_only=True, save_weights_only=False, verbose=0)

    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=numberEpochs, batch_size=batch,
                        verbose=0, callbacks=[mcp, stop_criteria])

    del model
    best_model = load_model("model_best.hdf5")

    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.xlim(0, numberEpochs)
    pyplot.title('Multivariates Modell: Classification (ACC) / ' + str(look_back) + ' / ' + str(forecast) + ' / '
                 + str(sequence) + ' / ' + str(numberEpochs) + ' / ' + str(batch))
    pyplot.legend()
    pyplot.show()

    # Model evaluieren
    trainScore = best_model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: ', trainScore)
    testScore = best_model.evaluate(testX, testY, verbose=0)
    print('Test Score: ', testScore)

    return best_model


# model1 = mlp_multivariate(13, 5, 1, 1000, 64)
# model1.save('mlp_reg_65421_80_20_1000_64.h5')

model2 = mlp_multivariate_trend(13, 1, 1, 1000, 64)
mlp_multivariate_trend(13, 1, 1, 1000, 64)
mlp_multivariate_trend(13, 1, 1, 1000, 64)
model2.save('mlp_class_76421_80_20_1000_64.h5')


def run_full_prediction():
    # Modellparameter
    look_back = 13
    forecast_arr = [1, 2, 3, 5, 10]
    sequence = [1, 5]
    numberEpochs = 1000
    batch = 64

    # Architektur der hidden layer (für die Ablage, manuell anpassen!)
    architecture = '65321L_80_20'

    for run in range(4):
        print("------------- / ", run, " / -------------")

        print("Regression / short and long term")
        for forecast in forecast_arr:
            print("Look-back: ", look_back, " / forecast: ", forecast, " / Sequence: ", sequence[0], " / Epochs: ",
                  numberEpochs, " / Batch-size: ", batch)
            model = mlp_multivariate(look_back, forecast, sequence[0], numberEpochs, batch)
            model.save(
                'mlp_reg_' + str(architecture) + '_' + str(look_back) + '_' + str(forecast) + '_' + str(sequence[0])
                + '_' + str(numberEpochs) + '_' + str(run) + '.h5')
            print("----------------------------")

        print("Regression / sequence")
        print("Look-back: ", look_back, " / forecast: ", forecast_arr[0], " / Sequence: ", sequence[1], " / Epochs: ",
              numberEpochs, " / Batch-size: ", batch)
        model = mlp_multivariate(look_back, forecast_arr[0], sequence[1], numberEpochs, batch)
        model.save(
            'mlp_reg_' + str(architecture) + '_' + str(look_back) + '_' + str(forecast_arr[0]) + '_' + str(sequence[1])
            + '_' + str(numberEpochs) + '_' + str(run) + '.h5')

        print("----------------------------")
        print("Classification")
        print("Look-back: ", look_back, " / forecast: ", forecast_arr[0], " / Sequence: ", sequence[0], " / Epochs: ",
              numberEpochs, " / Batch-size: ", batch)
        model = mlp_multivariate_trend(look_back, forecast_arr[0], sequence[0], numberEpochs, batch)
        model.save(
            'mlp_class_' + str(architecture) + '_' + str(look_back) + '_' + str(forecast_arr[0]) + '_' +
            str(sequence[0]) + '_' + str(numberEpochs) + '_' + str(run) + '.h5')


# run_full_prediction()

##################################################################################
##################################################################################
