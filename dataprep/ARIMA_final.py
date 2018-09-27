# ARIMA pt. 2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import seaborn as sns
import itertools
import sys


# Data Import

# DataFrame
df1 = pd.read_csv("BrentDataset_prep.csv")
df1 = df1.drop(columns=["Unnamed: 0", "Open", "High", "Low", "Volume", "Change"])
df1.index = df1["Date"]
df1 = df1.drop(columns=["Date"])
df1.index = pd.to_datetime(df1.index)

### Dayly mit ffill
df1_dayly = df1.copy()
df1_dayly = df1.asfreq("D")
df1_dayly = df1_dayly.fillna(method="ffill")


# DataFrame
df_dayly_df = df1_dayly.copy()
df_dayly_df.drop(columns=["Year", "Month", "Day"], inplace=True)

# DataFrame auf monatlicher Basis
df_monthly_copy = df_dayly_df.copy()
df_monthly_copy = df_monthly_copy.asfreq("MS")
# Nur Daten vom 01.01.2009 - 01.01.2018
df_monthly_copy = df_monthly_copy.loc["20090201" : "20180101"]



########################

# Split

########################


# Split (80 / 20)
y_train = df_monthly_copy[:96]
y_test = df_monthly_copy[96:]

#print(y_train.size)
#print(y_test.size)


########################

# Generiere das beste Modell

########################


# Deffiniere die p,d und q Parameter des ARIMA-Modells mit jedem Wert zwischen 0 und 2
p = d = q = range(0, 3)

# Generiere alle möglichen Kombinationen aus p,d,q
pdq = list(itertools.product(p, d, q))

# Generiere alle möglichen Kombinationen von saisonalen p,d,q Tripletten
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


# best_aic = np.inf
# best_pdq = None
# best_seasonal_pdq = None
# tmp_model = None
# best_mdl = None
#
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             tmp_mdl = sm.tsa.statespace.SARIMAX(y_train["Close"],
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=True,
#                                                 enforce_invertibility=True)
#             res = tmp_mdl.fit()
#             if res.aic < best_aic:
#                 best_aic = res.aic
#                 best_pdq = param
#                 best_seasonal_pdq = param_seasonal
#                 best_mdl = tmp_mdl
#         except:
#             print("Unexpected error:", sys.exc_info()[0])
#             continue
# print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


# 3. Best SARIMAX(0, 1, 0)x(0, 2, 1, 12)12 model - AIC:537.2446934869686 neu

########################

# Statistiken über das beste Modell

########################


# SARIMAX an die Daten anpassen
mdl = sm.tsa.statespace.SARIMAX(y_train["Close"],
                                order=(0, 1, 0),
                                seasonal_order=(0, 2, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
res = mdl.fit()

res.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
# plt.show()


########################

# Modell an die Daten anpassen

########################


res = sm.tsa.statespace.SARIMAX(y_train,
                                order=(0, 1, 0),
                                seasonal_order=(0, 2, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()

# Prediction von 02.2017 - 01.2018
pred = res.get_prediction(start=pd.to_datetime('2017-01-01'),
                          end=pd.to_datetime('2018-01-01'),
                          dynamic=True)

# Erstellen der Konfidenz Intervalle
pred_ci = pred.conf_int()

# Originaler Datensatz
y = df_monthly_copy.copy()

# Plotten der Predicition
# Originale Daten
ax = y.loc['20090201':].plot(label='Observed', color='#006699');

# Prediction mean
pred.predicted_mean.plot(ax=ax, label='One-step Ahead Prediction', color='#ff0066');

# Plotten des Konfidenzintervalls
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);

# Styling des Plots
# Hintergrund der Prediction wird Grau
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2017-02-01'), y.index[-1], alpha=.15, zorder=-1, color='grey');
# X Label
ax.set_xlabel('Date')
# Y Label
ax.set_ylabel('Close Price')
# Legende
plt.legend(loc='upper left')
plt.show()


# Werte der Prediction
y_hat = pred.predicted_mean
# Echte Werte ab dem 01.02.2017
y_true = y['20170201':]


# Mean Square Error als Gütekriterium
import math
mse = ((y_hat - y_true["Close"]) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
