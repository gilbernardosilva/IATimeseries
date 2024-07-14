
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import LSTM, Dense
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
    
        # Extract actual and predicted values
    y_actual = testProphet['y']
    y_pred = forecast['yhat']

    # Error Metrics Calculation
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    # Descriptive Statistics
    std_dev = np.std(y_actual)  
    mean = np.mean(y_actual)

    # Print Metrics
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Standard Deviation: {std_dev:.4f}')
    print(f'Mean: {mean:.4f}')

    return mse, rmse, mae, std_dev, mean


def lstm_prediction(train, test, target, regressors=[]):
    # Feature Selection
    features = [target] + regressors
    train_data = train[features]
    test_data = test[features]

    # Normalize Data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Prepare Sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :])
            y.append(data[i + seq_length, 0])  # Predict the target variable
        return np.array(X), np.array(y)

    seq_length = 10  # Number of previous time steps to use for prediction
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train Model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Reshape Predictions to 2D
    train_predict = train_predict.reshape(-1, 1) 
    test_predict = test_predict.reshape(-1, 1)  

    # Inverse Transform to Original Scale
    # Create empty arrays to fill with inverse transformed values
    train_predict = inverse_transform_with_regressors(train_predict, X_train, scaler, features)
    test_predict = inverse_transform_with_regressors(test_predict, X_test, scaler, features)

    # Plotting
    y_actual = test[target][seq_length:] # We start after the sequence length as we don't have predictions for them
    y_pred = test_predict

    # Error Metrics Calculation
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    # Descriptive Statistics
    std_dev = np.std(y_actual)  
    mean = np.mean(y_actual)

    # Print Metrics
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Standard Deviation: {std_dev:.4f}')
    print(f'Mean: {mean:.4f}')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test['Date_Time'][seq_length:], y_actual, label='Actual', marker='o', linestyle='-')  
    plt.plot(test['Date_Time'][seq_length:], y_pred, label='Predicted (LSTM)', marker='x', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.legend()
    plt.title(f'LSTM Prediction vs. Actual for {target}')
    plt.show()
    # Return the prediction
    return mse, rmse, mae, std_dev, mean


def inverse_transform_with_regressors(predictions, X, scaler, features):
        inv_predictions = []
        for i in range(len(predictions)):
            # Create a temporary array with the same shape as the original training data
            to_inverse = np.zeros((1, len(features)))
            to_inverse[0, 0] = predictions[i, 0]  # Fill in the prediction
            to_inverse[0, 1:] = X[i, -1, 1:]     # Fill in the regressors
            inv_predictions.append(scaler.inverse_transform(to_inverse)[0, 0])  # Get just the target column
        return np.array(inv_predictions)


def random_forest_prediction(train, test, target, regressors=[], lag_features=10): 
    features = [target] + regressors
    train_data = train[features]
    test_data = test[features]

    for i in range(1, lag_features + 1):
        train_data[f'{target}_lag_{i}'] = train_data[target].shift(i)
        test_data[f'{target}_lag_{i}'] = test_data[target].shift(i)

    # Drop Rows with NaN values due to shifting
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Separate Features (X) and Target (y)
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
   
    # Build Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42) 

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Error Metrics Calculation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    std_dev = np.std(y_test)
    mean = np.mean(y_test)

    # Print Metrics
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Standard Deviation: {std_dev:.4f}')
    print(f'Mean: {mean:.4f}')


    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Align time axis with test data
    test_plot_data = test.loc[X_test.index] 
    
    plt.plot(test_plot_data['Date_Time'], y_test, label='Actual', marker='o', linestyle='-')
    plt.plot(test_plot_data['Date_Time'], y_pred, label='Predicted (Random Forest)', marker='x', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.legend()
    plt.title(f'Random Forest Prediction vs. Actual for {target}')
    plt.show()

    # Return errors
    return mse, rmse, mae, std_dev, mean


def compare_models(train, test, targets, regressors_dict, seq_length=10, lag_features=10):
    results = {}

    for target in targets:
        regressors = regressors_dict.get(target, [])
        
        # Prophet
        mse, rmse, mae, std_dev, mean = prophet_prediction(train, test, target, regressors)
        results[f'Prophet_{target}'] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'STD_DEV': std_dev, 'MEAN': mean}

        # LSTM
        mse, rmse, mae, std_dev, mean = lstm_prediction(train, test, target, regressors)
        results[f'LSTM_{target}'] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'STD_DEV': std_dev, 'MEAN': mean}

        # Random Forest
        mse, rmse, mae, std_dev, mean = random_forest_prediction(train, test, target, regressors, lag_features=seq_length)  # Using seq_length here
        results[f'RandomForest_{target}'] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'STD_DEV': std_dev, 'MEAN': mean}

    # Create and Display Comparison Table
    results_df = pd.DataFrame(results).T
    print("\nError Metrics Comparison:")
    print(results_df)
    return results_df