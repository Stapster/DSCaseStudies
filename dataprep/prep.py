import csv
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
import numpy


csv_file = open("BrentDataset_Original.csv")
reader = csv.reader(csv_file)
#print(next(reader))

dta_v = pd.read_csv("BrentDataset_Original.csv")

def dta_vanilla ():
    return dta_v


def dta_preparation():
    dta = dta_v.reset_index()   #setzt Index zurück
    #print(dta.head())
    date_split = dta["index"].str.split(expand=True)    #neuer Datensatz aus Index, gesplittet in Monat und Tag
    del dta["index"]                                    #löscht Spalte Index aus ursprünglichem Datensatz
    dta = pd.concat((date_split, dta), axis=1)          #Verbindet beide Datensätze
    dta = dta.rename(columns={0: "Month", 1: "Day", "Date": "Year", "Price": "Close", "Vol.": "Volume", "Change %": "Change"})  #Umbennenungen
    dta.Change = [x.strip("%") for x in dta.Change]     #löscht %-Zeichen aus Spalte Change für jeden Datensatz
    dta.Volume = [x.replace("-","0") for x in dta.Volume]   #ersetzt "-"-Zeichen mit "0"
    dta[["Volume"]] = dta[["Volume"]].astype(float)         #Typanpassung
    dta[["Change"]] = dta[["Change"]].astype(float)         #Typanpassung
    dta[["Year"]] = dta[["Year"]].astype(str)               #Typanpassung
    dta["Date"] = dta["Year"] + "/" + dta["Month"] + "/" + dta["Day"]       #neuer Datensatz mit Spalte Date aus Year, Month und Day
    dta_date = pd.to_datetime(dta["Date"])              #Typanpassung für Date
    del dta["Date"]                             #löscht alte Spalte "Date"
    dta_final = pd.concat((dta, dta_date), axis=1)      #verbindet alten und neuen Datensatz
    dta_final.sort_values(by=["Date"], ascending=True, inplace=True)    #Sortierung der Daten nach Date
    #print(list(dta_final.columns.values))
    dta_final = dta_final[["Date","Year","Month","Day","Low","High","Open","Close","Change","Volume"]]  #finale Anordnung der Spalten
    #print(dta_final.head())
    return dta_final

def dta_rescaled(dta_final):
    #x = dta_final.values[:, 5:8]
    #y = dta_final[:, 7]
    #x = skp.MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
    scaler =skp.MinMaxScaler()
    dta_final[["Low", "High", "Open", "Close", "Change", "Volume"]] = scaler.fit_transform(dta_final[["Low", "High", "Open", "Close", "Change", "Volume"]])
    #x = skp.normalize(dta_final.values[:, 4:9])
    numpy.set_printoptions(precision=3)
    return dta_final

def plotdta (dta_final):
    plt.plot(dta_final.Date, dta_final.Close)
    #plt.hist(dta_final.Volume, normed=True, alpha=0.9)

print ("Before:")
print(dta_vanilla().head())
print(dta_vanilla().info())
print()
print("Clean data:")
print(dta_preparation().head())
print(dta_preparation().info())
print()
print("Rescaled data:")
print(dta_rescaled(dta_preparation()).head())


plt.show(plotdta(dta_preparation()))                        #Visualisierung Rohdaten
#plt.show(plotdta(dta_rescaled(dta_preparation())))             #Visualisierung skalierte Daten