# Bitte beachten:

# Im Ordner models sind die besten Modelle gespeichert und können mit keras.models.load_model wieder aufgerufen werden.
# Die Modelle sind nach dem Durchlauf, den Parametern und der Prediction benannt
# zu berücksichtigen sind primär die Modelle im Ordner "60_40" (vom data-split), entsprechend im Unterordner "best model"

# Beispiel: mlp_reg_76421L_13_1_1_1000_3.h5
# - mlp-Modell für eines der Regressionsprobleme
# - Modell-Architektur: 7/6/4/2/1, Verwendung von log-transformierten Daten
# - Die Prediction wird auf Basis der Werte aus den 13 vorangegangen Tagen durchgeführt, entsprechender Input wird benötigt
# - Es wird der Wert t+1 berechnet
# - Das Modell berechnet 1 Wert / nur bei der Sequenzprognose steht hier eine 5
# - Für das Training wurden 1000 Epochen angegeben
# - Dieses Modell wurde im 4. Durchlauf (counter beginnt bei 0) generiert