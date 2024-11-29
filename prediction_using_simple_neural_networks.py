import pandas as pd 
import numpy as np 
import netCDF4 as nc
from netCDF4 import Dataset

# visualization
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import seaborn as sns
import mpl_toolkits
%matplotlib inline

# machine learning 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.decomposition import PCA
from keras.layers import Dense, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
import tensorflow as tf
from sklearn import preprocessing
from keras import regularizers
from keras.layers import Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from statsmodels.tsa.arima_model import ARIMA

# other
import warnings
warnings.filterwarnings("ignore")
import datetime
import time
import math
# load the data
data_cuxhaven = pd.read_csv("Data/cuxhaven_de.csv")

# rename the columns
data_cuxhaven.columns = ['time', 'wind_u10', 'wind_v10', 'slp', 'weight', 'surge']
data_cuxhaven.head()
# plot the data
data_cuxhaven.plot(subplots=True, layout=(3, 2), figsize=(15,10))

plt.show()
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_percentage(data_cuxhaven)
def time_lag(data, lags):
    """
    Transforms the dataset to  a time series of grid information and spits back the time lagged time series
    data - the full name of the csv file
    """
    time_orig = pd.to_datetime('1900-01-01')

    df = pd.read_csv(data)
    df.columns = ['time', 'wind_u10', 'wind_v10', 'slp', 'weight', 'surge'] 
    
    # reorganize the matrix
    df_new = df.loc[df['weight'] == df['weight'].unique()[0]]
    df_new.drop(['weight'], axis = 1, inplace=True) #, 'surge'
    
    for i in range(1,10):
        df_sub = df.loc[df['weight'] == df['weight'].unique()[i]]
        df_sub.drop(['weight', 'surge'], axis = 1, inplace=True)
        df_new = pd.merge(df_new, df_sub, on='time')
    
    
    # lag the time series data
    lagged_df = df_new.copy() # to prevent modifying original matrix
    for j in range(lags):
        #lagged.drop(j, axis = 0, inplace = True)
        lagged_df['time'] = lagged_df['time'] + 6
        
        # remove the last row since there is no match for it in df_new
        lagged_df.drop(lagged_df.tail(1).index.item(), axis = 0, inplace = True)
        
        # remove the topmost row from df_new to match lagged
        df_new.drop(df_new.head(1).index.item(), axis = 0, inplace = True)
        
        # merge lagged data with df_new
        df_new = pd.merge(df_new, lagged_df, on = 'time', how = 'outer', \
                       suffixes = ('_left', '_right'))
    df_new = df_new.T.reset_index(drop=True).T
    ind = df_new.loc[pd.isna(df_new[df_new.shape[1]-1]), :].index
    df_new.drop(ind, inplace=True)
    
    # storm surge time series data
    surge_ts = pd.DataFrame(df.loc[df['weight'] == \
                                df['weight'].unique()[0]][['time', 'surge']])
    # remove missing/NaN values
    surge_ts.reset_index(inplace=True) # reset index for subsetting isnans
    surge_ts.drop(['index'], axis = 1, inplace=True)    
    indx = surge_ts.loc[pd.isna(surge_ts["surge"]), :].index
    df_new.drop(indx, inplace=True)
    surge_ts.drop(indx, inplace=True)
    
    # filter surge according to df_new
    lagged_time = list(df_new[0])
    time_df_new = [float(x) for x in df_new[0]]
    time_surge_ts = [float(x) for x in surge_ts['time']]
    time_both = []
    for k in lagged_time:
        if ((k in time_df_new) & (k in time_surge_ts)):
            time_both.append(int(k))
            
    surge_ts = surge_ts[surge_ts['time'].isin(time_both)]
    
    dt = pd.DataFrame(columns = ['date']);
    for i in surge_ts.index:
        dt.loc[i, 'date'] = time_orig + \
            datetime.timedelta(hours = int(surge_ts.loc[i, 'time']))
            
    surge_ts['date'] = dt
    df_new = df_new[df_new[0].isin([x*1.0 for x in time_both])]
    df_new.drop(4, axis = 1, inplace = True) # remove the un-lagged surge data
    return df_new, surge_ts
    data = 'Data/cuxhaven_de.csv'
x, surge = time_lag(data, 5)
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, surge, shuffle=False, test_size=0.20, random_state=42)
# visualize train and test values for reference
fig = plt.subplots(figsize=(15,8), dpi=300)
plt.plot(y_train.date, y_train.surge, label="train")
plt.plot(y_test.date, y_test.surge, label="test")
plt.title("Surge Height (m)", fontsize=16)
plt.xlabel('Time')
plt.ylabel('Surge Height (m)')
plt.legend(loc='best')
plt.show()
x_train.shape
x_test.shape
x.shape
surge.shape
y_train.shape
y_test.shape
x_norm_train = preprocessing.scale(x_train)
x_norm_test = preprocessing.scale(x_test)
print(x_norm_train)
print(y_test)
print(y_train)
def autcorrplt(data, lag, title, *args):
    """
    plots the autocorrelation of a pandas series object
    """
    acorr = [];
    for i in range(lag+1):
        acorr.append(data.autocorr(lag = i))
    
    plt.figure(figsize = (8, 6))
    plt.plot(acorr)
    plt.xlabel("Lag (in 6 hrs)", fontsize = 12)
    plt.ylabel("Correlation", fontsize = 12)
    plt.ylim([0, 1])
    plt.xlim([1, lag])
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(f"Autocorrelation in {title}", fontsize = 16, y=1.03)
    plt.savefig(f"Autocorrelation in {title}.png", dpi=300)
autcorrplt(data_cuxhaven['slp'], 30, "Sea Level Pressure")
autcorrplt(data_cuxhaven['wind_u10'], 30, "Wind U10")
autcorrplt(data_cuxhaven['wind_v10'], 30, "Wind V10")
autcorrplt(data_cuxhaven['surge'], 30, "Surge")
# build a function for implementing MLP Sequential Model 

def mlp_seq(x_norm_train, x_norm_test, y_train, y_test):
    """
    Builds mlp, trains and tests it
    """
    # simple model
    model = Sequential()
    model.add(Dropout(0.2, input_shape = (x_norm_train.shape[1],)))
    model.add(Dense(94, activation = 'sigmoid',\
                    input_shape = (x_norm_train.shape[1],)))
    model.add(Dense(94, activation='sigmoid'))
    model.add(BatchNormalization())
    # model.add(Dense(180, activation='relu'))
    model.add(Dense(1))
    
    # training model
    model.compile(loss = 'mean_squared_error', optimizer = 'adagrad', \
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    history = model.fit(x_norm_train, y_train['surge'], epochs = 50, \
              batch_size = 10, verbose = 1, validation_split=0.20)
    
    testPredict = model.predict(x_norm_test)
    
    # prepare the data for plotting
    y = y_test[:]
    y.reset_index(inplace=True)
    y.drop(['index'], axis = 1, inplace=True) 
    
    # model evaluation
    print()
    print("mse = ", mean_squared_error(y_test['surge'], testPredict))
    print("mae = ", mean_absolute_error(y_test['surge'], testPredict))
    print("r2_score = ", r2_score(y_test['surge'], testPredict))

    print()
    
    # plotting 
    sns.set_context('notebook', font_scale= 1.5)
    plt.figure(figsize=(20, 10))
    plt.plot(y_test['date'], y['surge'], color = 'blue')
    plt.plot(y_test['date'],testPredict, color= 'red')
    plt.legend(['Observed Surge', 'Predicted Surge'], fontsize = 14)
    plt.xlabel('Time')
    plt.ylabel('Surge Height (m)')
    plt.title("Observed vs. Predicted Storm Surge Height", fontsize=20, y=1.03)
    plt.savefig("1 mlp observed vs predicted surge height.png", dpi=300)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(y_test['surge'], testPredict, c='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.show()

    # list all the data in history
    print(history.history.keys())
    
    # summarize history for accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('MLP Model Accuracy', fontsize=18, y=1.03)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig("2 mlp model accuracy.png", dpi=300)
    plt.show()

    # summarize history for loss
    plt.figure(figsize = (12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MLP Model Loss', fontsize=18, y=1.03)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig("3 mlp model loss.png", dpi=300)
    plt.show()
    return testPredict
mlp_seq(x_norm_train, x_norm_test, y_train, y_test)
