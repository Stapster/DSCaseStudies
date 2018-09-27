import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 15,6

data = pd.read_csv("BrentDataset_prep.csv")

# Datetime als index
data.index = pd.to_datetime(data["Date"])
# Data schmälern
data = data.drop(columns=["Date", "Unnamed: 0", "Year", "Month", "Day"])

# Kontrolle
#print(data.index)
#print(data.head())

# Problem Frequency der DateTime ist None, das führt zu weiteren Problemen bei der Datenbearbeitung, also muss die Freuqency händisch eingesetzt werden, hier haben wir Frequency="B" weil Businesstag
data = data.asfreq("Q")
#print(data.index)
#print(data[data.isnull().any(axis=1)])
data_impute = data.fillna(method="ffill")


# Weitere Arbeit / Analysen erfolgen auf dem Close-Preis
ts = data_impute["Close"]
# Indizierung
#print(ts['2008-12-1':"2008-12-10"])
#print(ts["2008"])
#print(ts.loc[:, ts.isnull().any()])
#print(ts.index)

# Check for Stationarity of a Time Series
#plt.plot(ts)
#plt.show()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=20, center=False).mean()
    rolstd = timeseries.rolling(window=20, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


#test_stationarity(ts)
# Nach Dickey-Fuller Test ist unsere TimeSeries Non-Stationairy, für weitere berechnungen müssen wir sie Stationairy machen

# TimeSeries Stationairy machen
# Logarithmieren des TS für Verhältnismäßigkeiten
# Anschluss plotten des Moving Average auf die TS mit einem Zeitfenster von 20 Tagen (entspricht bei uns einem Monat, da nur Werkstage)
ts_log = np.log(ts)
moving_avg = ts_log.rolling(window=20, center=False).mean()
#plt.plot(moving_avg, color="red")
#plt.plot(ts_log)
#plt.show()

# Moving Average von TS abziehen
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_moving_avg_diff)

# Test mit expotentially moving average
expweighted_av = ts_log.ewm(halflife=20).mean()
#plt.plot(ts_log)
#plt.plot(expweighted_av, color="red")
#plt.show()

ts_log_expmoving_avg_diff = ts_log - expweighted_av
ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_expmoving_avg_diff)

# Elimanating Trend and Seasonality
ts_log_diff = ts_log - ts_log.shift()
#plt.plot(ts_log_diff)
#plt.show()
ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)


# Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


def decompose_plot(original, trend1, trend2, seasonal, residual1, residual2):
    # Original Plot
    plt.subplot(411)
    plt.plot(original, label='Original')
    plt.legend(loc='lower left')
    # Trend Plot
    plt.subplot(412)
    plt.plot(trend1, label='Trend1')
    plt.plot(trend2, color="red", label="Trend2")
    plt.legend(loc='lower left')
    # Seasonal Plot
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='lower left')
    # Residual PLot
    plt.subplot(414)
    plt.plot(residual1, label='Residuals1')
    plt.plot(residual2, color="red", label="Residuals2")
    #Optik
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

#print(decompose_plot(original=ts_log,trend1=trend,seasonal=seasonal,residual1=residual, residual2=[], trend2=[]))

# Stationarity Test der Residuals
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
#test_stationarity(ts_log_decompose)

# Schlussfolgerung: Wir haben vier mögliche Tests auf Stationarity gemacht, dabei schneidet ts_log_avg_diff/ts_log_decompose am besten ab.


# Vorhersage einer TimeSeries
# ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) Plots
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method="ols")

def acf_pcf(ts):
    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

#print(acf_pcf(ts_log_diff))

# p = 1 und q = 1

# Let's make an ARIMA (Auto Regressive Integrated Moving Avergae) model
from statsmodels.tsa.arima_model import ARIMA

def ARIMA_plot(ts):
    # AR Model
    model = ARIMA(ts_log, order=(1, 1, 0))
    results_AR = model.fit(disp=-1)
    plt.subplot(311)
    plt.plot(ts)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts)**2))


    # MA Model
    model = ARIMA(ts_log, order=(0, 1, 1))
    results_MA = model.fit(disp=-1)
    plt.subplot(312)
    plt.plot(ts)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts)**2))


    # Combined Model
    model = ARIMA(ts_log, order=(1, 1, 1))
    results_ARIMA = model.fit(disp=-1)
    plt.subplot(313)
    plt.plot(ts)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts)**2))
    plt.tight_layout()
    plt.show()

#print(ARIMA_plot(ts_log_diff))


 # Combined Model
model = ARIMA(ts_log, order=(1, 1, 1))
results_ARIMA = model.fit(disp=-1)

# Zurück zu normaler Skalierung
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
#print(predictions_ARIMA_diff.head()) #lag = 1 daher beginnt es nicht am ersten Business Day

# CumSum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print(predictions_ARIMA_diff_cumsum.head())

# Add to Base Number
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print(predictions_ARIMA_log.head(25))
#print(ts_log.head(25)) # Für den Vergleich

#diff_log = ts_log - predictions_ARIMA_log
#print(diff_log.head(25))

# Make a Prediction
predictions_ARIMA = np.exp(predictions_ARIMA_log)

diff = ts - predictions_ARIMA
#print(diff.head(25))


plt.plot(ts_log)
plt.plot(predictions_ARIMA_log)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()
