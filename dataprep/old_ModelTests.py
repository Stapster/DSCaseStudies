import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import RNN
import dataprep.DataAnalysis2 as source
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate

oil_prices = source.OilData()
oil_prices.calculate_avg()
oil_prices.normalize()

# print("// Read 'original' data vom Pandas-Dataframe ...")
# print(oil_prices.data_original.head())
# print("// Read transformed data vom Pandas-Dataframe ...")
# print(oil_prices.data.head())


def test_basic():
    # Create your first MLP in Keras
    # split into input (X) and output (Y) variables
    input_data = oil_prices.data.values[:, 0:6]
    output_data = oil_prices.data.values[:, 6]

    print(input_data.shape)
    print(input_data)

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=6, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(input_data, output_data, epochs=150, batch_size=10)
    # evaluate the model
    scores = model.evaluate(input_data, output_data)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def test_lstm():
    values = oil_prices.data.values

    # Split in Trainings- und Testdaten (hier ohne Validierung)
    train_test_ratio = 0.2
    split = int((values.size/7)*train_test_ratio)

    train = values[:split, :]
    test = values[split:, :]

    # Split in Input- und Target-Variablen
    train_x, train_y = train[:, 0:6], train[:, 6]
    test_x, test_y = test[:, 0:6], test[:, 6]

    # print(train_x.shape)
    # print(train_x)

    # print(train_y.shape)
    # print(train_y)

    # Reshape input in 3D - Format [samples, timesteps, features] für LSTM
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # Ab hier Copy Paste
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    # Werte in der Vorlage: 50 Epochs, batch size= 72, nur 1 50 LSTM

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, epochs=75, batch_size=40, validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
    inv_yhat = oil_prices.scaler_price.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
    inv_y = oil_prices.scaler_price.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    plt.plot(inv_y, label="original data")
    plt.plot(inv_yhat, label="prediction")
    plt.legend(loc="lower left")
    plt.show()


test_lstm()
