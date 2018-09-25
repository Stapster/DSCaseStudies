import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA


# Klasse zum Einlesen und bearbeiten der Daten
class OilData:

    def __init__(self):
        # Daten einlesen, überflüssige Spalten entfernen und Index festlegen
        self.data = pd.read_csv("BrentDataset_prep.csv")
        self.data.index = pd.to_datetime(self.data["Date"])
        self.data = self.data.drop(columns=["Date", "Unnamed: 0", "Year", "Month", "Day"])
        self.data_original = pd.read_csv("BrentDataset_prep.csv")
        self.data_original.index = pd.to_datetime(self.data_original["Date"])
        self.data_original = self.data_original.drop(columns=["Date", "Unnamed: 0", "Year", "Month", "Day"])
        self.scaler_price = MinMaxScaler()
        self.scaler_volume = MinMaxScaler()

    def calculate_avg(self):
        # Berechnet tägliche Durchschnittspreise anhand High und Low
        self.data["Avg"] = pd.DataFrame((self.data["High"] + self.data["Low"]) / 2.0, index=self.data.index)
        self.data_original["Avg"] = pd.DataFrame((self.data["High"] + self.data["Low"]) / 2.0, index=self.data.index)

        # Anpassung % auf dezimal -- ggf. schon vorher machen
        self.data["Change"] = self.data["Change"]/100

    def calculate_trend(self):
        # Berechnet den Trend als binäre Variable [0,1] = [fallen, steigen] anhand der change-Werte
        change = self.data["Change"]
        c_arr = []
        for p in change:
            if p >= 0:
                c_arr.append(1)
            else:
                c_arr.append(0)

        self.data["Trend"] = pd.DataFrame(c_arr, index=self.data.index)
        self.data_original["Trend"] = pd.DataFrame(c_arr, index=self.data.index)

    def normalize(self):
        # Skalierung der Daten / fit_transform und getrennte Transformation wäre auch denkbar
        self.scaler_price.fit(self.data["Avg"].values.reshape(-1, 1))

        self.data[["Low", "High", "Open", "Close", "Avg"]] = \
            self.scaler_price.transform(self.data[["Low", "High", "Open", "Close", "Avg"]])

        # Idee hinter dem 2. Scaler: Anderer Wertebereich, getrennte Transformation macht hier mehr Sinn
        self.scaler_volume.fit(self.data["Volume"].values.reshape(-1, 1))
        self.data["Volume"] = self.scaler_volume.transform(self.data["Volume"].values.reshape(-1, 1))

        # print("// Transformed Data")
        # print(self.data[["Low", "High", "Open", "Close", "Volume", "Avg"]].head())


def pricerange_analysis(inputdata):
    # Analyse der Preisveränderung
    range_prices = inputdata["High"]-inputdata["Low"]
    plt.title("Price-Ranges")
    plt.plot(range_prices)
    plt.show()


def smooth_volume(inputdata):
    # Marktvolumenentwicklung glätten und anzeigen
    mavg_vol = inputdata["Volume"].rolling(window=40, center=False).mean()
    plt.title("Market-Volume (moving AVG 40d lag)")
    plt.plot(inputdata["Avg"], label="Avg-Price")
    plt.plot(mavg_vol, label="Volume")
    plt.legend(loc="lower left")
    plt.show()


def decomposition(inputdata):
    # Dekomposition
    # Notiz: ExpMovingAvg mit 5 Tagen lag sieht fast wie AVG mit 10 aus, was besser ist muss später geprüft werden

    # Moving Average
    movingavg = inputdata["Avg"].rolling(window=10, center=False).mean()
    movingavg_diff = inputdata["Avg"] - movingavg

    # Exponential Weighted Moving Average
    ewa = inputdata["Avg"].ewm(halflife=10).mean()
    ewa_diff = inputdata["Avg"] - ewa
    ewa.dropna(inplace=True)

    # Plot zum Vergleich
    plt.subplot(311)
    plt.plot(inputdata["Avg"], label='Original')
    plt.legend(loc='lower left')
    plt.title("Original-Data")
    plt.subplot(312)
    plt.plot(ewa, label='EXP')
    plt.plot(movingavg, color="red", label="AVG")
    plt.plot(inputdata["Avg"], color="grey", label="Original")
    plt.legend(loc='lower left')
    plt.title("Trend (lag 10)")
    plt.subplot(313)
    plt.plot(ewa_diff, label='EXP')
    plt.plot(movingavg_diff, color="red", label="AVG")
    plt.legend(loc='lower left')
    plt.title("Residuals (lag 10)")
    plt.tight_layout()
    plt.show()


def arima_test(inputclass):
    # ARIMA-Modell aus DataAnalysis.py
    # TODO Fehler finden und beheben
    data = inputclass.data
    data_original = inputclass.data_original
    scaler_price = inputclass.scaler_price

    movingavg = data["Avg"].rolling(window=10, center=False).mean()
    movingavg_diff = data["Avg"] - movingavg

    model = ARIMA(data["Avg"], order=(2, 1, 2))
    results_arima = model.fit(disp=-1)
    plt.plot(movingavg_diff)
    plt.plot(results_arima.fittedvalues, color='red')
    plt.title('RSS: %.4f' % sum((results_arima.fittedvalues-movingavg_diff)**2))
    # plt.show()

    predictions_arima_diff = pd.Series(results_arima.fittedvalues, copy=True)
    predictions_arima_diff_cumsum = predictions_arima_diff.cumsum()
    predictions_arima_scaled = pd.Series(data["Avg"].ix[0], index=data["Avg"].index)
    predictions_arima_scaled = predictions_arima_scaled.add(predictions_arima_diff_cumsum, fill_value=0)
    predictions_arima = scaler_price.inverse_transform(predictions_arima_scaled.values.reshape(-1, 1))
    predictions_arima = pd.DataFrame(predictions_arima, index=data.index)

    plt.title("ARIMA-Fit")
    plt.plot(predictions_arima)
    plt.plot(data_original, color="grey")
    # plt.show()


def run_all_tests():
    oilprices = OilData()

    pricerange_analysis(oilprices.data)

    oilprices.calculate_avg()
    oilprices.normalize()

    smooth_volume(oilprices.data)

    decomposition(oilprices.data)
    arima_test(oilprices)


# run_all_tests()
