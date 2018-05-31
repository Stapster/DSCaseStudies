import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split

# % Change wid noch nicht importiert wegen einem Formatierungsproblem
# Year-Spalte in der .csv eingefügt, da der Import sonst Müll wird
# TODO Formatierungsproblem %Change beheben
# Date-Year-Merge hinbekommen --> Fertig
# TODO Missing Data !!!
dataImport = pd.read_csv("BrentDataset.csv", usecols=[0, 1, 2, 3, 4, 5, 6])

# Merge Date and Year and convert it to Pandas' datetime-format
dataImport['DateYear'] = dataImport['Date'] + ' ' + dataImport['Year'].map(str)
dataImport['DateYear'] = pd.to_datetime(dataImport['DateYear'])
dataImport = dataImport.drop(columns=['Date', 'Year'])
dataImport = dataImport.drop(index=0)   # Deleting missing value entry

# Moving DateYear to the first column again
cols = dataImport.columns.tolist()
cols = cols[-1:] + cols[:-1]
dataImport = dataImport[cols]

# Reverse data set / we want it starting in the past and moving towards the present
# brauchen wir vllt nicht, müssen wir später bei den indize prüfen im Modell
dataImport = dataImport.iloc[::-1]

print(dataImport.head())

# fehlender Wert gefunden, Datensatz manuell aus csv gelöscht
# geht bestimmt auch eleganter
for vol in dataImport.values:
    if vol[5] == '-':
        print(vol, ' --> Missing Value ... delete entry!')

'''Scaling Data to [0,1] interval
only for the float-values, date is treated seperately'''
# TODO Scaler funktioniert noch nicht ohne reshape ??
# TODO wie bekomme ich das Datum wieder vernünftig in das DataFrame?
scaler = MinMaxScaler(feature_range=(0, 1))
dateYear = dataImport['DateYear']
scaledData = dataImport.drop(columns=['DateYear'])
scaledData = scaler.fit_transform(scaledData.values.reshape(len(scaledData.values), len(scaledData.columns)))
print(scaledData)
