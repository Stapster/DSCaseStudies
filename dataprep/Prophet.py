### Stock market Prediction with Prophet
### -> http://pythondata.com/stock-market-forecasting-with-prophet/
### -> http://pythondata.com/forecasting-time-series-data-with-prophet-part-1/
### -> http://pythondata.com/forecasting-time-series-data-with-prophet-part-2/



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
import pystan

### Data Prep
df = pd.read_csv("BrentDataset_prep.csv")
df.reset_index()
#df.index = pd.to_datetime(df["Date"])
df = df.drop(columns=["Unnamed: 0", "Year", "Month", "Day"])
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
df_dt = df.copy()
#df_dt.index = pd.to_datetime(df["Date"])
df_dt["Mid Price"] = (df_dt['Low']+df_dt['High'])/2.0
#df_dt.drop(columns=["Date"], inplace=True)

# Vorbereitung für Prophet
dp = pd.DataFrame({"ds": df_dt["Date"], "y": df_dt["Close"]})
dp.reset_index(drop=True)

# Logarithmierung der y um es Stationary zu machen
dp["y_org"] = dp["y"]
dp["y"] = np.log(dp["y"])

# Modelling
model = Prophet(daily_seasonality=True)
model.fit(dp);

