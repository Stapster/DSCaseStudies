import matplotlib.pylab as plt
import dataprep.DataAnalysis2 as source

oil_prices = source.OilData()
oil_prices.calculate_avg()
oil_prices.calculate_trend()

dataset = oil_prices.data_original

def test_indicators():
    oil_prices.obv()
    oil_prices.pvt()
    oil_prices.rsi()
    oil_prices.macd()
    dataset = oil_prices.data_original
    dataset = dataset[["Date","Close","Low","High","Open","Change","Volume","Avg","Trend", "obv", "pvt", "rsi", "macd_val", "macd_signal_line"]]
    dataset = dataset.set_index("Date")
    print(dataset.info())
    print(dataset[["Close","Volume", "obv", "pvt", "rsi", "macd_val", "macd_signal_line"]].iloc[2400: , 0:6])
    print("\n" + "Mean:")
    print(dataset.mean())
    plt.plot(dataset.rsi)
    plt.show()

test_indicators()