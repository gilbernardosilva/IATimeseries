
import numpy as np
from datetime import datetime
import pandas as pd
import copy
from neuralprophet import NeuralProphet
from matplotlib import pyplot


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

    print(data['Date_Time'])

    return data



def DataAnalyis(data,plot_cols):
    data1 = copy.copy(data)
    data1.index = data1['Date_Time']
    plot_features = data1[plot_cols]
    _ = plot_features.plot(subplots=True)

    pyplot.show()



def forecast_with_neuralprophet(data, target, test_size=0.2):
    """
    Performs time-series forecasting using NeuralProphet.

    Args:
        data (pd.DataFrame): The input DataFrame containing time-series data.
        target (str): The name of the column to be predicted.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.

    Returns:
        pd.DataFrame: The DataFrame containing actual and predicted values.
    """

    # Ensure 'Date_Time' column is datetime type and set as index
    data["Date_Time"] = pd.to_datetime(data["Date_Time"])
    data.set_index("Date_Time", inplace=True)

    # Split data into training and test sets
    train_size = int(len(data) * (1 - test_size))
    train = data.iloc[:train_size].copy().reset_index()
    test = data.iloc[train_size:].copy().reset_index()

    # Rename columns to match NeuralProphet's format
    train = train.rename(columns={"Date_Time": "ds", target: "y"})
    test = test.rename(columns={"Date_Time": "ds", target: "y"})

    # Create and fit the NeuralProphet model
    model = NeuralProphet(
        n_forecasts=len(test),
        n_lags=24,  # Adjust based on your data and needs
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    metrics = model.fit(train, freq="H")  # Assuming hourly data

    # Predict on the test set and combine results
    future = model.make_future_dataframe(test, n_historic_predictions=True)
    forecast = model.predict(future)
    forecast = forecast[['ds', 'y', 'yhat1']]  # Extract relevant columns

    # Prepare final DataFrame with actual and predicted values
    results = forecast.merge(test[['ds', 'y']], on='ds', how='left')
    results.rename(columns={'yhat1': 'Predicted', 'y': 'Actual'}, inplace=True)

    # Plot actual vs. predicted values
    pyplot.figure(figsize=(12, 8))
    pyplot.plot(results['ds'], results['Actual'], label='Actual')
    pyplot.plot(results['ds'], results['Predicted'], label='Predicted')
    pyplot.title(f'Forecasting {target} with NeuralProphet')
    pyplot.xlabel('Date_Time')
    pyplot.ylabel(target)
    pyplot.legend()
    pyplot.show()
    
    
    #Optional Evaluation
    from neuralprophet import set_random_seed

    set_random_seed(0)
    
    # Use cross-validation to assess prediction performance
    df_cv = cross_validation(model, n_lags=24, epochs=50, disable_tqdm=True)
    # Get the performance metrics for each prediction
    df_p = performance_metrics(df_cv)

    print(df_p.head().to_markdown(numalign='left', stralign='left'))
    from neuralprophet import plot_cross_validation_metric
    fig_cv = plot_cross_validation_metric(df_cv, metric='MAE')
    
    return results