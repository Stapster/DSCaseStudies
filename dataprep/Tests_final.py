### Übersicht Statistischer Tests


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import seaborn as sns

########################

# Data Prep

########################

# Series

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

########################

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

### Weekly (Sonntag letzter Tag)
df1_weekly = df1_dayly.copy()
df1_weekly = df1_weekly.asfreq("W")

### Monthly(Letzter Tag des Monats)
df1_monthly = df1_dayly.copy()
df1_monthly = df1_monthly.asfreq("M")


########################

# Data Analysis

########################


# Untersuchen von Linearer Abhängigkeit mit einem Scatterplot

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
        corr = X.corr().values[0][1]
        ax.set_ylabel('Original');
        ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
        ax.set_aspect('equal');

        # top and right spine from plot
        sns.despine();

    fig.tight_layout()
    plt.show()

#lin_dep(df_dayly)


########################
########################


# Autocorrelation Function (ACF), Partial Correlation Function (PCF), QQ-Plot, Histogramm

def ts_plot(y, lags=None, title=''):
    # Für die Weiterverarbeitung braucht es die Daten als Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Berechnung von Rolling Mean / Rolling STD
    rolling_mean = y.rolling(window=12).mean()
    rolling_std = y.rolling(window=12).std()

    # Plot Layout
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # Time Series Plot
    y.plot(ax=ts_ax, label="Data")
    # Rolling Mean Plot
    rolling_mean.plot(ax=ts_ax, color='crimson', label="Rolling Mean");
    # Rolling STD Plot
    rolling_std.plot(ax=ts_ax, color='darkslateblue', label="Rolling STD");
    ts_ax.legend(loc="best")
    ts_ax.set_title(title);

    # ACF und PACF
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # QQ Plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('Normal QQ Plot')

    # Hist Plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    plt.show()
    return

#ts_plot(df_dayly, 100, title="Dayly Daten mit einem Lag=100") # Grundsätzliche Daten mit einem Lag von 100


# Zum Vergleich: Random Walk Plot

# Random Seed für die Wiederholbarkeit
np.random.seed(2)

# Parameter initiieren
T = 1000
e = np.random.normal(size=T)
x = np.zeros_like(e)

# Simulation des Random Walk
for t in range(T):
    x[t] = x[t - 1] + e[t]

# Random Walk plotten
# ts_plot(x, lags=100, title='Random Walk')


########################

# Data Composition

########################


# Trend Composition

def trend(data):

    y = data

    # Layout Auslegen
    fig, axes = plt.subplots(2, 2, sharey=False, sharex=False);
    fig.set_figwidth(16);
    fig.set_figheight(8);

    # Oben Links
    axes[0][0].plot(y.index, y, label='Original');
    axes[0][0].plot(y.index, y.rolling(window=5).mean(), label='5-Day Rolling Mean', color='crimson');
    axes[0][0].set_xlabel("Years");
    axes[0][0].set_ylabel("Close - Price");
    axes[0][0].set_title("5-Day Moving Average");
    axes[0][0].legend(loc='best');

    # Oben rechts
    axes[0][1].plot(y.index, y, label='Original')
    axes[0][1].plot(y.index, y.rolling(window=10).mean(), label='10-Day Rolling Mean', color='crimson');
    axes[0][1].set_xlabel("Years");
    axes[0][1].set_ylabel("Close - Price");
    axes[0][1].set_title("10-Day Moving Average");
    axes[0][1].legend(loc='best');

    # Unten Links
    axes[1][0].plot(y.index, y, label='Original');
    axes[1][0].plot(y.index, y.rolling(window=30).mean(), label='30-Day Rolling Mean', color='crimson');
    axes[1][0].set_xlabel("Years");
    axes[1][0].set_ylabel("Close - Price");
    axes[1][0].set_title("30-Day Moving Average");
    axes[1][0].legend(loc='best');

    # Unten Rechts
    axes[1][1].plot(y.index, y, label='Original');
    axes[1][1].plot(y.index, y.rolling(window=100).mean(), label='100-Day Rolling Mean', color='crimson');
    axes[1][1].set_xlabel("Years");
    axes[1][1].set_ylabel("Close - Price");
    axes[1][1].set_title("100-Day Moving Average");
    axes[1][1].legend(loc='best');
    plt.tight_layout();
    plt.show()

# trend(df_dayly)


########################
########################


# Seasonal Composition (Monatlich)

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


########################
########################


# Vanilla Seasonal Decompose
# Seasonal Decompose function als freq die jeweilige Frequency als "" einfügen

def decompose(freq):
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Frequency setzen
    df_decomp = df_dayly.copy()
    df_decomp = df_decomp.asfreq(freq)

    # Model
    decomp = seasonal_decompose(df_decomp, model="additive")

    observed = decomp.observed
    trend = decomp.trend
    seasonal = decomp.seasonal
    residuals = decomp.resid

    # Plotting
    plt.subplot(411)
    plt.plot(observed, color="b")
    plt.title("Observed")
    plt.subplot(412)
    plt.plot(trend, color="g")
    plt.title("Trend")
    plt.subplot(413)
    plt.plot(seasonal, color="r")
    plt.title("Seasonal")
    plt.subplot(414)
    plt.plot(residuals, color="y")
    plt.title("Residuals")
    plt.tight_layout()
    plt.show()

# decompose("W")


########################
########################


### Test auf Stationarity via Augumented Dickey Fuller Test

def dick(data):
        from statsmodels.tsa.stattools import adfuller

        print('Results of Dickey-Fuller Test (Normal):')
        dftest = adfuller(data, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

# Ohne Transformation
#dick(df_dayly)

# First order Differencing Transformation
df_dayly_diff = np.diff(df_dayly)
#dick(df_dayly_diff)
# ts_plot(df_dayly_diff, 100, title="Transformierte (Differencing) Dayly Daten mit einem Lag=100")


