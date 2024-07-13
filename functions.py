
import numpy as np
from datetime import datetime
import pandas as pd
import copy
import matplotlib.pyplot as plt
from prophet import Prophet


def clean_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Number of lines before deletion of null values ", len(data))
    data.dropna(inplace=True)
    print("Number of lines before deletion of duplicate values ", len(data))
    data.drop_duplicates(inplace=True)
    print("Final number of lines ", len(data))
    return data


def extract_datetime(data, column_name='Time'): 
    data['Date_Time'] = pd.to_datetime(data[column_name]) 
    data = data.drop(columns=[column_name])
    return data



def data_analysis(data,plot_cols):
    data1 = copy.copy(data)
    data1.index = data1['Date_Time']
    plot_features = data1[plot_cols]
    _ = plot_features.plot(subplots=True)


   

def divide_data(percentageOfTest, data):
    print("Original Dataset Size", len(data))
    lengTrain = round(len(data) * percentageOfTest)
    DivideTrain = data[:-lengTrain]
    DivideTest = data.drop(DivideTrain.index)
    print("Train Dataset Size", len(DivideTrain))
    print("Test Dataset Size", len(DivideTest))
    return DivideTrain, DivideTest


def prophet_prediction(train, test, target, regressors=[]):
    fitting = train[['Date_Time', target] + regressors]
    fitting.columns = ['ds', 'y'] + regressors

    m = Prophet(changepoint_prior_scale=0.01,
    changepoint_range=0.8)
    
    m.add_seasonality(name='17h',period=0.7, fourier_order=30)

    for regressor in regressors:
        m.add_regressor(regressor)
    
    m.fit(fitting)

    testProphet = m.make_future_dataframe(periods=300, freq='H')
  
    
    testProphet = test[['Date_Time', target] + regressors]
    testProphet.columns = ['ds', 'y'] + regressors
    
    forecast = m.predict(testProphet)
    plt.figure(figsize=(12, 8))
    plt.plot(testProphet['ds'], testProphet['y'], label='Actual')
    plt.plot(testProphet['ds'], forecast['yhat'], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.legend()
    plt.show()
    
    return forecast



# def forProphet2(train, test, target, regressors=[]):
#     fitting = train[['Date_Time', target] + regressors]
#     fitting.columns = ['ds', 'y'] + regressors

#     m = Prophet(interval_width=0.95)
#     numberofhours = 17
#     hours_day = 24
#     period_value = numberofhours / hours_day
#     print(period_value)
#     m.add_seasonality(name='17h', period=period_value*5, fourier_order=30)

#     for regressor in regressors:
#         m.add_regressor(regressor)
    
#     m.fit(fitting)
    
#     testProphet = test[['Date_Time', target] + regressors]
#     testProphet.columns = ['ds', 'y'] + regressors
    
#     forecast = m.predict(testProphet)
    
#     plt.figure(figsize=(12, 8))
#     plt.plot(testProphet['ds'], testProphet['y'], label='Actual')
#     plt.plot(testProphet['ds'], forecast['yhat'], label='Predicted')
#     plt.xlabel('Time')
#     plt.ylabel(target)
#     plt.legend()
#     plt.show()
    
#     return forecast
