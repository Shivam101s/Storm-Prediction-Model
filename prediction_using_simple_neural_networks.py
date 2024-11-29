# Required Libraries
import pandas as pd
import numpy as np
from netCDF4 import Dataset

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
%matplotlib inline

# Machine Learning Libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from statsmodels.tsa.api import ARIMA

# General Utilities
import warnings
warnings.filterwarnings("ignore")
import datetime

# Load Data
data_path = "Data/cuxhaven_de.csv"
data_cuxhaven = pd.read_csv(data_path)

# Rename Columns
data_cuxhaven.columns = ['time', 'wind_u10', 'wind_v10', 'slp', 'weight', 'surge']

# Plot Initial Data
data_cuxhaven.plot(subplots=True, layout=(3, 2), figsize=(15, 10))
plt.show()

# Missing Data Check
def missing_percentage(df):
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({'Total': total, 'Percent': percent.round(2)})

print(missing_percentage(data_cuxhaven))

# Time Lag Function
def time_lag(data_path, lags):
    time_orig = pd.to_datetime('1900-01-01')
    df = pd.read_csv(data_path)
    df.columns = ['time', 'wind_u10', 'wind_v10', 'slp', 'weight', 'surge']

    # Restructure Data
    df_new = df[df['weight'] == df['weight'].unique()[0]].drop(['weight'], axis=1)
    for i in range(1, 10):
        df_sub = df[df['weight'] == df['weight'].unique()[i]].drop(['weight', 'surge'], axis=1)
        df_new = pd.merge(df_new, df_sub, on='time')

    # Add Lags
    for j in range(lags):
        lagged_df = df_new.copy()
        lagged_df['time'] += 6
        lagged_df = lagged_df[:-1]  # Trim last row
        df_new = pd.merge(df_new[:-1], lagged_df, on='time', suffixes=('', f'_lag{j+1}'))

    # Align Data with Surge Time Series
    surge_ts = df[df['weight'] == df['weight'].unique()[0]][['time', 'surge']].dropna()
    surge_ts['date'] = [time_orig + datetime.timedelta(hours=int(round(t))) for t in surge_ts['time']]
    valid_times = set(df_new['time']) & set(surge_ts['time'])
    surge_ts = surge_ts[surge_ts['time'].isin(valid_times)]
    df_new = df_new[df_new['time'].isin(valid_times)].drop(columns=['surge'])

    return df_new, surge_ts

# Generate Time Lagged Data
x, surge = time_lag(data_path, 5)
x_train, x_test, y_train, y_test = train_test_split(x, surge, test_size=0.2, shuffle=False, random_state=42)

# Normalize Data
x_norm_train = pd.DataFrame(preprocessing.scale(x_train.drop(columns=['time'])), columns=x_train.drop(columns=['time']).columns)
x_norm_test = pd.DataFrame(preprocessing.scale(x_test.drop(columns=['time'])), columns=x_test.drop(columns=['time']).columns)

# Autocorrelation Plot Function
def autcorrplt(data, lag, title):
    data = pd.Series(data) if not isinstance(data, pd.Series) else data
    acorr = [data.autocorr(lag=i) for i in range(lag+1)]
    plt.figure(figsize=(8, 6))
    plt.plot(acorr)
    plt.xlabel("Lag (6 hrs)", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.ylim([0, 1])
    plt.title(f"Autocorrelation in {title}", fontsize=16)
    plt.show()

# Autocorrelation Plots
autcorrplt(data_cuxhaven['slp'], 30, "Sea Level Pressure")
autcorrplt(data_cuxhaven['wind_u10'], 30, "Wind U10")
autcorrplt(data_cuxhaven['wind_v10'], 30, "Wind V10")
autcorrplt(data_cuxhaven['surge'], 30, "Surge")

# MLP Model Function
def mlp_seq(x_norm_train, x_norm_test, y_train, y_test):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(x_norm_train.shape[1],)))
    model.add(Dense(94, activation='sigmoid'))
    model.add(Dense(94, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])

    history = model.fit(x_norm_train, y_train['surge'], epochs=50, batch_size=10, validation_split=0.2, verbose=1)

    testPredict = model.predict(x_norm_test)

    # Evaluation
    print("MSE:", mean_squared_error(y_test['surge'], testPredict))
    print("MAE:", mean_absolute_error(y_test['surge'], testPredict))
    print("R2 Score:", r2_score(y_test['surge'], testPredict))

    # Plot Observed vs Predicted
    plt.figure(figsize=(20, 10))
    plt.plot(y_test['date'], y_test['surge'], label='Observed Surge', color='blue')
    plt.plot(y_test['date'], testPredict, label='Predicted Surge', color='red')
    plt.legend(fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Surge Height (m)')
    plt.title("Observed vs. Predicted Storm Surge Height", fontsize=20)
    plt.show()

    return model, history

# Train MLP Model
mlp_seq(x_norm_train, x_norm_test, y_train, y_test)
