import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA

# TODO Data Split

data = pd.read_csv("BrentDataset_prep.csv")
data.index = pd.to_datetime(data["Date"])   # Datetime als index
data = data.drop(columns=["Date", "Unnamed: 0", "Year", "Month", "Day"])

# Einfache Analysen mit Close-Preis
#closeData = data["Close"]
# Indizierung
#print(closeData['2010-11-1':"2010-11-15"])
#print(closeData[:"2008-02-10"])
#print(closeData["2008"])
#high_prices = data.loc[:, "High"].values
#low_prices = data.loc[:, "Low"].values
#mid_prices = (high_prices+low_prices)/2.0
#split = int(mid_prices.size*0.8)
#train_data = mid_prices[:split]
#test_data = mid_prices[split:]

# Analyse der Preisveränderung
range_prices = data["High"]-data["Low"]
plt.title("Price-Ranges")
plt.plot(range_prices)
plt.show()

# Berechnung des Durchschnittspreises pro Tag
data["Avg"] = pd.DataFrame((data["High"]+data["Low"])/2.0, index=data.index)
data_original = data["Avg"]

# Anpassung % auf dezimal
data["Change"] = data["Change"]/100

# Skalierung der Daten: fit_transform denkbar
scaler_price = MinMaxScaler()
scaler_price.fit(data["Avg"].values.reshape(-1, 1))
data[["Low", "High", "Open", "Close", "Change", "Avg"]] = \
    scaler_price.transform(data[["Low", "High", "Open", "Close", "Change", "Avg"]])

# Idee hinter dem 2. Scaler: Anderer Datentyp, getrennte Transformation macht ggf. Sinn
scaler_volume = MinMaxScaler()
scaler_volume.fit(data["Volume"].values.reshape(-1, 1))
data["Volume"] = scaler_volume.transform(data["Volume"].values.reshape(-1, 1))

print("// Transformed Data")
print(data[["Low", "High", "Open", "Close", "Change", "Volume", "Avg"]].head())

# Glättung Marktvolumen
mAvg_vol = data["Volume"].rolling(window=20, center=False).mean()
plt.title("Market-Volume (moving AVG 20d lag)")
plt.plot(data["Avg"], label="Avg-Price")
plt.plot(mAvg_vol, label="Volume")
plt.legend(loc="lower left")
plt.show()

# Tests aus DataAnalysis.py
movingAvg = data["Avg"].rolling(window=10, center=False).mean()
movingAvg_diff = data["Avg"] - movingAvg

ewa = data["Avg"].ewm(halflife=10).mean()
ewa_diff = data["Avg"] - ewa
ewa.dropna(inplace=True)

# Vergleich Moving AVG Exp Moving AVG
plt.title("Moving AVG vs EXP-Moving AVG")
plt.plot(movingAvg, label="MAVG", color="green")
plt.plot(ewa, label="EXP-MAVG", color="red")
plt.plot(data["Avg"], color="grey")
plt.legend(loc="lower left")
plt.show()

# Dekomposition
# Notiz: ExpMovingAvg mit 5 Tagen lag sieht fast wie AVG mit 10 aus, was besser ist muss später geprüft werden
plt.subplot(311)
plt.plot(data["Avg"], label='Original')
plt.legend(loc='lower left')
plt.title("Original-Data")
plt.subplot(312)
plt.plot(ewa, label='EXP')
plt.plot(movingAvg, color="red", label="AVG")
plt.plot(data["Avg"], color="grey", label="Original")
plt.legend(loc='lower left')
plt.title("Trend (lag 10)")
plt.subplot(313)
plt.plot(ewa_diff, label='EXP')
plt.plot(movingAvg_diff, color="red", label="AVG")
plt.legend(loc='lower left')
plt.title("Residuals (lag 10)")
plt.tight_layout()
plt.show()

#ARIMA-Modell aus DataAnalysis.py

model = ARIMA(data["Avg"], order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(movingAvg_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-movingAvg_diff)**2))
plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_scaled = pd.Series(data["Avg"].ix[0], index=data["Avg"].index)
predictions_ARIMA_scaled = predictions_ARIMA_scaled.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = scaler_price.inverse_transform(predictions_ARIMA_scaled.values.reshape(-1, 1))
predictions_ARIMA = pd.DataFrame(predictions_ARIMA, index=data.index)

plt.title("ARIMA-Fit")
plt.plot(predictions_ARIMA)
plt.plot(data_original, color="grey")
plt.show()
