### Übersicht Statistischer Tests

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import seaborn as sns

# Data Prep als Series
df = pd.read_csv("BrentDataset_prep.csv")
#df = df.drop(columns=["Unnamed: 0", "Year", "Month", "Day", "Open", "High", "Low", "Volume", "Change"])
df.index = pd.to_datetime(df["Date"])
df = df.drop(columns=["Date"])
df.index = pd.to_datetime(df.index)
df = df["Close"]
df.columns = ["Close Price"]


### Dayly mit ffill
df_dayly = df.copy()
df_dayly = df.asfreq("D")
df_dayly = df_dayly.fillna(method="ffill")


### Weekly (Sonntag letzter Tag)
df_weekly = df_dayly.copy()
df_weekly = df_weekly.asfreq("W")

### Monthly(Letzter Tag des Monats)
df_monthly = df_dayly.copy()
df_monthly = df_monthly.asfreq("M")





# Data Prep als DF
df1 = pd.read_csv("BrentDataset_prep.csv")
df1 = df1.drop(columns=["Unnamed: 0", "Open", "High", "Low", "Volume", "Change"])
df1.index = df1["Date"]
df1 = df1.drop(columns=["Date"])
df1.index = pd.to_datetime(df1.index)


### Dayly mit ffill
df1_dayly = df1.copy()
df1_dayly = df1.asfreq("D")
df1_dayly = df1_dayly.fillna(method="ffill")


### Weekly (Sonntag letzter Tag)
df1_weekly = df1_dayly.copy()
df1_weekly = df1_weekly.asfreq("W")

### Monthly(Letzter Tag des Monats)
df1_monthly = df1_dayly.copy()
df1_monthly = df1_monthly.asfreq("M")





# Untersuchen von Linearer Abhängigkeit mit einem Scatterplot
### --> http://dacatay.com/data-science/part-1-time-series-basics-python/

def lin_dep(data):


    ncols = 3
    nrows = 3
    lags = 9

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 6 * nrows))
    for ax, lag in zip(axes.flat, np.arange(1, lags + 1, 1)):
        lag_str = 't-{}'.format(lag)
        X = (pd.concat([data, data.shift(-lag)], axis=1, keys=['y'] + [lag_str]).dropna())

        # plot data
        X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
        corr = X.corr().values()[0][1]
        ax.set_ylabel('Original');
        ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
        ax.set_aspect('equal');

        # top and right spine from plot
        sns.despine();

    fig.tight_layout()
    plt.show()

#lin_dep(df_dayly)


# Wir haben überall einen serielle Korrelation in unseren Daten.
# Wenn wir dies nicht in Betrach ziehen, wird der Standardfehler unserer Koeffizienten unterschätzt.







# Autocorrelation Function (ACF) und Partial Correlation Function (PCF)
def ts_plot(y, lags=None, title=''):
    '''
    Calculate acf, pacf, histogram, and qq plot for a given time series
    '''
    # if time series is not a Series object, make it so
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # initialize figure and axes
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # time series plot
    y.plot(ax=ts_ax)
    plt.legend(loc='best')
    ts_ax.set_title(title);

    # acf and pacf
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('Normal QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    plt.show()
    return

# ts_plot(df_dayly, 10, title="Dayly Daten mit einem Lag=10") # Grundsätzliche Daten



# Es lässt sich an diesem Plot feststellen, dass es die gleiche Struktur eines Random Walks hat
# Damit ist dieser Datensatz im Grunde nicht genau vorhersagbar
# Das liegt daran, dass unser Datensatz non-stationary ist.


### Test auf Stationarity via Augumented Dickey Fuller Test
from statsmodels.tsa.stattools import adfuller

def dickey(data, window):
    from statsmodels.tsa.stattools import adfuller

    print('Results of Dickey-Fuller Test (Normal):')
    dftest = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    # Stationär machen durch das Abziehen des Moving Average
    moving_avg = data.rolling(window=window, center=False).mean()
    diff = data - moving_avg
    diff.dropna(inplace=True)

    print('Results of Dickey-Fuller Test (Nach Korrektur):')
    dftest = adfuller(diff, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


#dickey(df_dayly, 10)
# auch hier lässt sich feststelle, dass df_dayly nicht-stationary ist.
# die diff von dayly und dem moving average hingegen schon!



# Daten Stationary machen, nach Dickey Fuller-Test
# Gibt hierfür noch andere Methoden
moving_avg = df_dayly.rolling(window=10, center=False).mean()
diff = df_dayly - moving_avg
diff.dropna(inplace=True)

#ts_plot(diff, 10, title="Stationary Data Output with Lag=10")






### Trend Composition

def trend(data):

    y = data

    # define figure and axes
    fig, axes = plt.subplots(2, 2, sharey=False, sharex=False);
    fig.set_figwidth(14);
    fig.set_figheight(8);

    # push data to each ax
    # upper left
    axes[0][0].plot(y.index, y, label='Original');
    axes[0][0].plot(y.index, y.rolling(window=5).mean(), label='5-Day Rolling Mean', color='crimson');
    axes[0][0].set_xlabel("Years");
    axes[0][0].set_ylabel("Close - Price");
    axes[0][0].set_title("5-Day Moving Average");
    axes[0][0].legend(loc='best');

    # upper right
    axes[0][1].plot(y.index, y, label='Original')
    axes[0][1].plot(y.index, y.rolling(window=10).mean(), label='10-Day Rolling Mean', color='crimson');
    axes[0][1].set_xlabel("Years");
    axes[0][1].set_ylabel("Close - Price");
    axes[0][1].set_title("10-Day Moving Average");
    axes[0][1].legend(loc='best');

    # lower left
    axes[1][0].plot(y.index, y, label='Original');
    axes[1][0].plot(y.index, y.rolling(window=30).mean(), label='30-Day Rolling Mean', color='crimson');
    axes[1][0].set_xlabel("Years");
    axes[1][0].set_ylabel("Close - Price");
    axes[1][0].set_title("30-Day Moving Average");
    axes[1][0].legend(loc='best');

    # lower right
    axes[1][1].plot(y.index, y, label='Original');
    axes[1][1].plot(y.index, y.rolling(window=100).mean(), label='100-Day Rolling Mean', color='crimson');
    axes[1][1].set_xlabel("Years");
    axes[1][1].set_ylabel("Close - Price");
    axes[1][1].set_title("100-Day Moving Average");
    axes[1][1].legend(loc='best');
    plt.tight_layout();
    plt.show()

#trend(df_dayly)



# Seasonal Composition


def seasonal(data):
    # Achsen Namen
    month_names = pd.date_range(start='2008-01-01', periods=12, freq='MS').strftime('%b')

    # Pivotieren des DF
    df_piv_box = data.pivot(index='Year', columns='Month', values='Close')

    # Reindizierung der Pivot-DF mit 'month_names'
    df_piv_box = df_piv_box.reindex(columns=month_names)

    # Boxplot bauen
    fig, ax = plt.subplots();
    df_piv_box.plot(ax=ax, kind='box');
    ax.set_title('Seasonal Effect per Month', fontsize=24);
    ax.set_xlabel('Month');
    ax.set_ylabel('Close Price');
    ax.xaxis.set_ticks_position('bottom');
    fig.tight_layout();
    plt.show()

# seasonal(df1_monthly)

# Leichte Saisonalität lässt sich feststellen finde ich



### Test auf Seasonality
### --> https://machinelearningmastery.com/time-series-seasonality-with-python/

# nach Decomposing der Daten ist ersichtlich, dass es eine saisonale Komponente gibt --> siehe dazu decompose Output in ARIMA


