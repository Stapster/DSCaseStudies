### ARIMA (Autoregressive Integrated Moving average)
### -> https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
### -> https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c
### -> ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPythonDraft.pdf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Data Prep
df = pd.read_csv("BrentDataset_prep.csv")
#df = df.drop(columns=["Unnamed: 0", "Year", "Month", "Day", "Open", "High", "Low", "Volume", "Change"])
df.index = pd.to_datetime(df["Date"])
df = df.drop(columns=["Date"])
df.index = pd.to_datetime(df.index)
df = df["Close"]
df.columns = ["Close Price"]


### Dayly mit ffill
df_dayly = df.copy()
df_dayly = df.asfreq("D")
df_dayly = df_dayly.fillna(method="ffill")


### Weekly (Sonntag letzter Tag)
df_weekly = df_dayly.copy()
df_weekly = df_weekly.asfreq("W")

### Monthly(Letzter Tag des Monats)
df_monthly = df_dayly.copy()
df_monthly = df_monthly.asfreq("M")



# Seasonal Decompose
# Seasonal Decompose function als freq die jeweilige Frequency als "" einfpgen
def decompose(freq):
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Frequency setzen
    df_decomp = df_dayly.copy()
    df_decomp = df_decomp.asfreq(freq)

    # Model
    decomp = seasonal_decompose(df_decomp, model="additive")

    observed = decomp.observed
    trend = decomp.trend
    seasonal = decomp.seasonal
    residuals = decomp.resid

    # Plotting
    plt.subplot(411)
    plt.plot(observed, color="b")
    plt.title("Observed")
    plt.subplot(412)
    plt.plot(trend, color="g")
    plt.title("Trend")
    plt.subplot(413)
    plt.plot(seasonal, color="r")
    plt.title("Seasonal")
    plt.subplot(414)
    plt.plot(residuals, color="y")
    plt.title("Residuals")
    plt.suptitle(freq + " - " + "Decomposition")
    plt.tight_layout()
    plt.show()

decompose("W")

###  Autocorrelation
def autocorr(freq):
    from pandas.plotting import autocorrelation_plot

    # Frequency setzen
    df_autocorr = df_dayly.copy()
    df_autocorr = df_autocorr.asfreq(freq)

    # Model
    autocorrelation_plot(df_autocorr)

    # Plot
    plt.show()

# autocorr("M") # lag = 18
# autocorr("W") # lag = 90
# autocorr("B") # lag = 480
# autocorr("D") # lag = 690



### ARIMA
def arima(freq, lag):
    from statsmodels.tsa.arima_model import ARIMA

    # Frequency setzen
    df_arima = df_dayly.copy()
    df_arima = df_arima.asfreq(freq)

    # Model
    model = ARIMA(df_arima, order=(lag, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Plot Residual Errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind="kde")
    plt.show()
    print(residuals.describe())

#arima("M", 18) # Monthly ARIMA-Model


# # Prediction
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error
#
# X = df_monthly.values
#
# size = int(len(X) * 0.66) # 2/3 & 1/3 Split
# # Def. Train / Test
# train, test = X[0:size], X[size:len(X)]
#
# # Set History, to keep track of observations
# history = [x for x in train]
#
# # Make a Prediction
# predictions = list()
# for t in range(len(test)):
#     model = ARIMA(history, order = (18,1,0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print("predicted=%f, expected=%f" % (yhat, obs))
# error = mean_squared_error(test, predictions)
# print("Test MSE: %.3f" % error)
#
# # Plot
# plt.plot(test)
# plt.plot(predictions, color="red")
# plt.show()





###  Autocorrelation

def autocorr(freq):
    from pandas.plotting import autocorrelation_plot

    # Frequency setzen
    df_autocorr = df_dayly.copy()
    df_autocorr = df_autocorr.asfreq(freq)

    # Model
    autocorrelation_plot(df_autocorr)

    # Plot
    plt.show()


# autocorr("D")