import pandas as pd
import numpy as np
import numpy
import math
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Convolution1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras import losses
import dataprep.DataAnalysis2 as source
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from scipy.stats import boxcox
import datetime as dt
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay

holidays = pd.read_csv("holidays.csv", sep=";")
print(holidays)
weekmask = "Mon Tue Wed Thu Fri"

bday = CustomBusinessDay(holidays=holidays["Date"], weekmask=weekmask)          #freq C
#dtime = datetime(2013, 12, 31)
#print((dtime + 1 * bday))


oil_prices = source.OilData()
oil_prices.calculate_avg()
oil_prices.calculate_trend()
# oil_prices.normalize()

print("Datenbasis:")
print(oil_prices.data.head())

print("\n"+"FBProphet:")

#Datenanbindung und Darstellung

dataset_orig = oil_prices.data_original

#plt.plot(dataset_orig.Date, dataset_orig.Close)
#plt.show()

#Datenbasis für FBProphet
data_prophet = dataset_orig.rename(columns={"Close": "y"})
data_prophet["ds"] = data_prophet["Date"]
data_prophet = data_prophet.set_index("Date")
data_prophet = data_prophet.drop(columns=["Low", "High", "Open", "Change", "Volume", "Avg"])

#Datenbasis log-Transformiert
data_prophet_log = data_prophet.copy()
data_prophet_log["y"] = np.log(data_prophet_log["y"])

#Datenpartitionierung
database_split = data_prophet       #EINGABE

train_size = int(len(database_split) * 0.7)         #2583*0,7
test_size = len(database_split) - train_size
train, test = database_split.iloc[0:train_size, :], database_split.iloc[train_size:len(database_split), :]
train_log, test_log = data_prophet_log.iloc[0:train_size, :], data_prophet_log.iloc[train_size:len(database_split), :]

print("\n"+"Observations: %d" % (len(database_split)))
print("\n"+"Training Observations: %d" % (len(train)))
print("Training-Tail:")
print(train.tail(3))
print("\n"+"Testing Observations: %d" % (len(test)))
print("Test-Head:")
print(test.head(3))
################################
#Basis-Datensatz hier ändern!!!#
################################




def prophet():

    data = train                        #EINGABE
    periods = test_size

    #Modell -train
    model = Prophet()           #optional weekly_seasonality=False
    #model.add_seasonality(name="yearly", period=365, fourier_order=12, mode="multiplicative")
    model.fit(data)

    # Essentiell: Frequency "B" für Business Days, problem -> holidays werden berechnet
    # periods: Int number of periods to forecast forward.
    #
    future_data = model.make_future_dataframe(periods=periods, freq="C")

    print("Result of make_future_dataframe:")
    print(future_data.tail())

    forecast_data = model.predict(future_data)
    #print(forecast_data[["yhat", "yhat_lower", "yhat_upper"]].tail())

    #forecast_data["yhat"] = np.exp(forecast_data["yhat"])
    #forecast_data["yhat_lower"] = np.exp(forecast_data["yhat_lower"])
    #forecast_data["yhat_upper"] = np.exp(forecast_data["yhat_upper"])

    full_data = forecast_data.set_index("ds").join(database_split.set_index("ds"))
    #full_data["y"] = np.exp(full_data["y"])

    full_data = full_data.dropna()                          #löscht predictions für holidays die für y missing values erzeugen
    full_data["Change_Pred"] = full_data["yhat"].diff()
    full_data["Trend_Pred"] = np.where(full_data["Change_Pred"] >= 0, 1, 0)
    full_data["Trend"] = full_data["Trend"].astype(int)            #ursprünglich fehler weil missing values (y wg. holidays)
    print("\n"+"Observed vs. Predicted Data:")
    print(full_data[["y", "yhat", "Trend", "Trend_Pred"]].head())  #, "yhat_lower", "yhat_upper" .iloc[50:150, :]

    #full_data[["yhat","y"]].plot()
    full_data_stat = (full_data.yhat - full_data.y)
    print("\n"+"Prediction-Statistics (yhat-y):")
    print(full_data_stat.describe())
    # print(full_data.info())
    # zeigt MVs
    # print([full_data[full_data["y"].isnull()][full_data.columns[full_data.isnull().any()]]])

    print("\n"+"Accuracy:")
    print(full_data[["Trend", "Trend_Pred"]].iloc[train_size:train_size + periods])
    print(accuracy_score(full_data.Trend[train_size:train_size + periods].values, full_data.Trend_Pred[train_size:train_size + periods].values))

    # model.plot(forecast_data, xlabel="Date", ylabel="Price")
    # plt.show()
    model.plot_components(forecast_data)
    plt.show()
    # print(full_data.Trend.values)
    #print(full_data.Trend_Pred.values)

    fig, ax1 = plt.subplots()
    ax1.plot(full_data.y)
    ax1.plot(full_data.yhat)
    ax1.plot(full_data.yhat_upper, color='black', linestyle=':', alpha=0.5)
    ax1.plot(full_data.yhat_lower, color='black', linestyle=':', alpha=0.5)
    ax1.set_title('Crude Oil Price (Brent): Actual (Blue) vs. Forecast (Orange)')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    plt.show()

    """
    print("Test-Kopf:")
    print(test.head())

    yhat = np.exp(forecast_data["yhat"])    #"yhat"][1730:1739]
    y = np.exp(test["y"])    #.loc[1730:1739]
    #y = np.exp(test["y"])
    #print(yhat)
    mse = ((yhat - y) ** 2).mean()
    print("Prognosedaten:")
    #print(yhat)
    #print("Testdaten:")
    #print(y)
    #print(mse)
    print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
    """
def prophet_crossval():

    #Model -whole dataset with cv

    data = data_prophet
    periods = 110        # Abstand zwischen forecasts der Länge horizon

    m = Prophet()  # optional growth="logistic"
    m.fit(data)

    #future_data = m.make_future_dataframe(periods=test_size, freq="B")  # Essentiell: Frequency "B" für Business Days
    #forecast_data = m.predict(future_data)
    #forecast_data["yhat"] = np.exp(forecast_data["yhat"])
    #forecast_data["yhat_lower"] = np.exp(forecast_data["yhat_lower"])
    #forecast_data["yhat_upper"] = np.exp(forecast_data["yhat_upper"])
    #print(forecast_data.tail())

    # Derzeit im Standard nur Cross-Validation über days und nicht über customdays (siehe bday) möglich
    # Anzahl Tage: 3653
    df_cv = cross_validation(m, initial="{} days".format(int(3653*0.7)), period="{} days".format(periods), horizon="{} days".format(int(3653*0.03)))    #default: initial=730; horizon=365

    #df_cv = cross_validation(m, initial="{:.2f} days".format(train_size), period="500 days", horizon="{:.2f} days".format(test_size))            # pandas timedelta



    #df_cv["y"] = np.exp(df_cv["y"])
    #df_cv["yhat"] = np.exp(df_cv["yhat"])
    #df_cv["yhat_lower"] = np.exp(df_cv["yhat_lower"])
    #df_cv["yhat_upper"] = np.exp(df_cv["yhat_upper"])

    # df_cv['mape'] = np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])
    # mean absolute percent error, by horizon
    # mape = df_cv.groupby('horizon', as_index=False).aggregate({'mape': 'mean'})

    # mape.head()

    print(df_cv)
    #df_cv = df_cv.dropna()  # löscht predictions für holidays die für y missing values erzeugen
    #df_cv["Change_Pred"] = df_cv["yhat"].diff()
    #df_cv["Trend_Pred"] = np.where(df_cv["Change_Pred"] >= 0, 1, 0)
    #df_cv["Trend"] = df_cv["Trend"].astype(int)


                                                   #output: ds,yhat,yhat_lower,yhat_upper,y,cutoff
    df_p = performance_metrics(df_cv, rolling_window = 1/int(3653*0.03))
    print(df_p)

    plot_cross_validation_metric(df_cv, metric="mse")      #Bug in plot.py gefixt
    plt.show()



    """
    ax1 = plt.subplots()
    ax1.plot(df_cv.rmse)
    ax1.plot(df_cv.mae)
    ax1.set_title('Cross-Validation-Performance')
    ax1.set_ylabel('rmse/mae')
    ax1.set_xlabel('Horizon')
    plt.show()
    """

# prophet()
# prophet_crossval()