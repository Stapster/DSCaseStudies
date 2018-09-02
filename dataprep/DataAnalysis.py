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

# Weitere Arbeit / Analysen erfolgen auf dem Close-Preis
ts = data["Close"]
# Indizierung
#print(ts['2008-12-1':"2008-12-10"])
#print(ts["2008"])

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

# Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log, model="additive", freq=20)
#trend = decomposition.trend
#seasonal = decomposition.seasonal
#residual = decomposition.resid



plt.subplot(311)
plt.plot(ts_log, label='Original')
plt.legend(loc='lower left')
plt.subplot(312)
plt.plot(expweighted_av, label='Trend_EXP')
plt.plot(moving_avg, color="red", label="Trend_AVG")
plt.legend(loc='lower left')
plt.subplot(313)
plt.plot(ts_log_expmoving_avg_diff, label='Residuals_EXP')
plt.plot(ts_log_moving_avg_diff, color="red", label="Residuals_AVG")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

