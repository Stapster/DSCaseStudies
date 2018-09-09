import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
import talib


df = pd.read_csv("BrentDataset_prep.csv")

# Datetime als index
#df.index = pd.to_datetime(df["Date"])
#df.reset_index()
# Data schmälern
df = df.drop(columns=["Unnamed: 0", "Year", "Month", "Day"])
# Daten sortieren
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Kontrolle
#print(df.head())


# Allgemeine Betrachtung der Daten (Data Exploring)

# Hinzufügen von Dateindex (für die X-Achse)
df_dt = df
df_dt.index = pd.to_datetime(df["Date"])

# Neue Spalte Mid-Prices
df_dt["Mid Price"] = (df_dt['Low']+df_dt['High'])/2.0

# Plotten des Mid-Prices
#plt.figure(figsize = (18,9))
#plt.plot(df_dt['Mid Price'])
#plt.ylabel('Mid Price',fontsize=18)
#plt.xlabel("Date", fontsize=18)
#plt.show()

# Change des Mid-Prices
MidPrice_Change = df_dt["Mid Price"].pct_change() # Berechnen des 1 Tages Unterschiedes des Mid-Prices
MidPrice_Change = MidPrice_Change.dropna()
#plt.hist(MidPrice_Change, bins=50)
#plt.xlabel("Mid-Price 1-Day Percent Change")
#plt.show()

# Korrelationen mit Pearson (Normalverteilte Daten durch Eyeballing des Histogramms)
# Neue Spalte: 5-Tage Zukunftspreis, also wie wird der Preis in 5 Tagen sein

def price_corr(x):

    df_dt[str(int(x)) + "D Future Mid"] = df_dt["Mid Price"].shift(-int(x))
    # Neue Spalten: 5-Tage Zukunftspreis zu Current: Unterschied
    df_dt[str(int(x)) + "D Mid-Future pct"] =df_dt[str(int(x)) + "D Future Mid"].pct_change(int(x))
    df_dt[str(int(x)) + "D Mid pct"] = df_dt["Mid Price"].pct_change(x)
    # Korrelation-Matrix zwischen dem 5-Day PCT Change von Current und Future
    corr = df_dt[[str(int(x)) + "D Mid-Future pct", str(int(x)) + "D Mid pct"]].corr()
    print(corr)

    # Visualisierung der Korrelation
    plt.scatter(df_dt[str(int(x)) + "D Mid-Future pct"], df_dt[str(int(x)) + "D Mid pct"])
    plt.show()

#price_corr(20) # mittelfristiger aufwärtstrend
#price_corr(2) # kurzfristiger aufwärtstrend
#price_corr(100) # langfristig mean-reversed (Preis jumped)


# an dieser Stelle könnten wir noch weitere Analysen fahren, wie bspw. Volatilität, aber die Sinnhaftigkeit sollten wir immer im Auge behalten.

# Kurze Analysen des Brent Preises
df_dt["SMA100"] = talib.SMA(df_dt["Close"], timeperiod=100) # Simple Moving Average
df_dt["RSI100"] = talib.RSI(df_dt["Close"], timeperiod=100) # Relative String Index (when close to zero, price is due to rebounce, when close to 1 price is due to decline)

# Können als Feature verwendet werden um mit KNN weiterzumachen, hier aber die Analyse durch LSTM


# Berechnung des mittleren Preises am Tag durch den Durchschnitt von High / Low
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices+low_prices)/2.0

#print(mid_prices.size) # 2583 DataPoints

# Index resetten
df.reset_index(inplace=True, drop=True)

# Aufteilen in Test- und Traningsdaten, hier 80 (Training) / 20 (Test) Split
split = int(mid_prices.size*0.9)
train_data = mid_prices[:split] # 2453 DataPoints
test_data = mid_prices[split:] # 259 DataPoints



# Skalierung der Daten zwischen 0 / 1 -> hier kann man auch nochmal andere Möglichkeiten in Betracht ziehen
# Normalisiere BEIDE Daten mit Respekt zu den TrainingsDaten, da ich keinen Zugang zu den Test-Daten habe!
# Reshaping ist notwendig für den Scaler
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)


# Trainiere den Scaler (MinMaxScaler) mit den Trainigs-Daten
smoothing_window_size = 500 # Window Size, ich trainiere die Normalisierung der Daten in Fenster, nicht zu klein! Da jeder WindwoWechsel einen Break in der Normalisierung darstellt
for di in range(0,2000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# Normalisiere das letzte Teilstück 2066 - 2000
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# Reshape TrainingDate
train_data = train_data.reshape(-1)

# Normalisiere TestDaten mit TrainingsDaten Scaler und Reshape auf normal
test_data = scaler.transform(test_data)
test_data = test_data.reshape(-1)

plt.plot(test_data)
plt.show()


# Daten werden gesmoothed / gemittelt mit dem Exponential Moving Average (wird eher bei Ölpreisen verwendet) --> könnten aber auch an dieser Stelle überlegen den einfachen MA zu nehmen, wird beides gemacht und miteinander vergleicht.
# Führt dazu, dass die Daten smoother sich dem Verlauf des MidPrises annähern
# Nur auf den Trainingsdaten
EMA = 0.0
gamma = 0.1 # Smoothing Factor zwischen null und eins, gamma = 1 Vorhersagewert ist gleich dem Messwert (keine Glättung), für 0 Vorhersage bleibt unverändert (Glättung zu einer Parallelen zur X-Achse)
for ti in range(2324):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Für Testen und Visualisierung
all_mid_data = np.concatenate([train_data,test_data],axis=0)


# Standard Average
window_size = 20 # 1% batches
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors))) # MSE: 0.00242

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()

