import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler

from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 15,6

class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step


data = pd.read_csv("BrentDataset_prep.csv")
data.index = pd.to_datetime(data["Date"])   # Datetime als index
data = data.drop(columns=["Date", "Unnamed: 0", "Year", "Month", "Day"])

# Kontrolle
#print(data.index)
#print(data.head())

closeData = data["Close"]                          # Einfache Analysen mit Close-Preis
# Indizierung
#print(closeData['2010-11-1':"2010-11-15"])
#print(closeData[:"2008-02-10"])
#print(closeData["2008"])

high_prices = data.loc[:, "High"].values
low_prices = data.loc[:, "Low"].values
mid_prices = (high_prices+low_prices)/2.0

split = int(mid_prices.size*0.8)
train_data = mid_prices[:split]
test_data = mid_prices[split:]

#range_prices = high_prices-low_prices
#plt.plot(range_prices)
#plt.show()

scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

scaler.fit(train_data)
train_data_tf = scaler.transform(train_data).reshape(-1)
test_data_tf = scaler.transform(test_data).reshape(-1)

# aus DataAnalysis zum Testen

#ewa = train_data.ewm(halflife=20).mean()
#ewa_diff = train_data - ewa
#ewa.dropna(inplace=True)
data["Avg"] = pd.DataFrame(mid_prices, index=data.index)
#print(data)

