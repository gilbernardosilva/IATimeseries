import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from prophet import Prophet
import copy


def clean_data(data):
    """
    Cleans the input DataFrame by handling infinite values, missing values, and duplicate rows.

    Parameters:
    data (DataFrame): The input data to be cleaned.

    Returns:
    DataFrame: The cleaned data.
    """
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Number of lines before deletion of null values ", len(data))

    # Drop rows with any NaN values
    data.dropna(inplace=True)
    print("Number of lines before deletion of duplicate values ", len(data))

    # Drop duplicate rows
    data.drop_duplicates(inplace=True)
    print("Final number of lines ", len(data))

    return data


def extract_datetime(data, column_name="Time"):
    """
    Extracts datetime from a specified column and creates a new 'Date_Time' column.

    Parameters:
    data (DataFrame): The input data containing the datetime column.
    column_name (str): The name of the column to convert to datetime. Default is 'Time'.

    Returns:
    DataFrame: The data with the new 'Date_Time' column and the original datetime column dropped.
    """
    # Convert the specified column to datetime
    data["Date_Time"] = pd.to_datetime(data[column_name])

    # Drop the original datetime column
    data = data.drop(columns=[column_name])

    return data


def data_analysis(data, plot_cols):
    """
    Performs data analysis by plotting specified columns over time.

    Parameters:
    data (DataFrame): The input data containing the datetime and other columns to be analyzed.
    plot_cols (list): List of column names to be plotted.

    Returns:
    None
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    data1 = copy.copy(data)

    # Set the index to the 'Date_Time' column
    data1.index = data1["Date_Time"]

    # Select the columns to plot
    plot_features = data1[plot_cols]

    # Plot the selected features
    _ = plot_features.plot(subplots=True)


def divide_data(percentage, data):
    """
    Divides the data into training and testing sets based on the specified percentage for the test set.

    Parameters:
    percentage (float): The percentage of data to be used as the test set.
    data (DataFrame): The input data to be divided.

    Returns:
    tuple: A tuple containing the training set and the test set DataFrames.
    """
    print("Original Dataset Size", len(data))

    # Calculate the number of rows for the test set
    lengTrain = round(len(data) * percentage)

    # Divide the data into training and test sets
    DivideTrain = data[:-lengTrain]
    DivideTest = data.drop(DivideTrain.index)

    print("Train Dataset Size", len(DivideTrain))
    print("Test Dataset Size", len(DivideTest))

    return DivideTrain, DivideTest


def prophet_prediction(train, test, target, regressors=[]):
    """
    Performs time series forecasting using the Prophet library.

    Args:
    - train (DataFrame): Training dataset containing Date_Time, target, and regressors.
    - test (DataFrame): Test dataset containing Date_Time, target, and regressors.
    - target (str): Name of the target variable to forecast.
    - regressors (list, optional): List of additional regressor variables. Defaults to an empty list.

    Returns:
    - tuple: Mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE),
             standard deviation of actual values, and mean of actual values.
    """

    # Prepare data for Prophet
    fitting = train[["Date_Time", target] + regressors]
    fitting.columns = ["ds", "y"] + regressors

    # Initialize Prophet model
    m = Prophet(changepoint_prior_scale=0.01, changepoint_range=0.8)

    # Add seasonality and regressors to the model
    m.add_seasonality(name="17h", period=0.7, fourier_order=30)
    for regressor in regressors:
        m.add_regressor(regressor)

    # Fit the model to the training data
    m.fit(fitting)

    # Prepare test data for forecasting
    testProphet = test[["Date_Time", target] + regressors]
    testProphet.columns = ["ds", "y"] + regressors

    # Make forecasts
    forecast = m.predict(testProphet)

    # Plotting actual vs. predicted values
    plt.figure(figsize=(12, 8))
    plt.plot(testProphet["ds"], testProphet["y"], label="Actual")
    plt.plot(testProphet["ds"], forecast["yhat"], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel(target)
    plt.legend()
    plt.title(f"Prophet Prediction vs. Actual for {target}")
    plt.show()

    # Extract actual and predicted values
    y_actual = testProphet["y"]
    y_pred = forecast["yhat"]

    # Error Metrics Calculation
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    # Descriptive Statistics
    std_dev = np.std(y_actual)
    mean = np.mean(y_actual)

    # Print Metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Mean: {mean:.4f}")

    return mse, rmse, mae, std_dev, mean


def lstm_prediction(train, test, target, seq_length, regressors=[], save_model=False):
    """
    Performs time series forecasting using an LSTM neural network.

    Args:
        train (DataFrame): 
            - The training dataset containing the historical data used to train the LSTM model. 
            - This DataFrame should include columns for the target variable you want to predict (`target`) 
              and any optional regressor variables (`regressors`).

        test (DataFrame):
            - The testing dataset used to evaluate the performance of the trained LSTM model. 
            - Similar to `train`, this DataFrame should also include columns for the target variable and regressors.

        target (str): 
            - The name of the column in your DataFrames (`train` and `test`) that represents the variable you want to forecast. 
            - This is the variable the LSTM will learn to predict.

        seq_length (int): 
            - The length of input sequences used to train the LSTM.
            - Each sequence consists of `seq_length` consecutive time steps of the target and regressor variables.
            - The LSTM learns to predict the next value in the sequence based on these past values.

        regressors (list, optional): 
            - A list of column names in your DataFrames that represent additional variables (regressors) 
              that you believe might influence the target variable's behavior.
            - These regressors are included as input features to the LSTM model alongside the target variable.

        save_model (bool, optional): 
            - A flag to indicate whether or not you want to save the trained LSTM model after training.
            - Defaults to False (the model is not saved). If set to True, the model is saved to the 'models' directory.

    Returns:
        - tuple: A tuple containing the following performance metrics:
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - std_dev: Standard Deviation of the actual target values in the test set
            - mean: Mean of the actual target values in the test set
    """

    # Feature Selection and Data Normalization
    features = [target] + regressors
    train_data = train[features]
    test_data = test[features]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Prepare Sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length, :])
            y.append(data[i + seq_length, 0])  # Predict the target variable
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Build LSTM Model
    model = Sequential()
    model.add(
        LSTM(50, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2]))
    )
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train LSTM Model
    model.fit(
        X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)
    )

    # Make Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

        # Save Model (If Specified)
    if save_model:
        # Create a "models" directory if it doesn't exist
        os.makedirs('models', exist_ok=True)  

        # Save the model to the "models" directory
        model_path = os.path.join('models', 'lstm_model_' + target + '.h5')
        model.save(model_path)
        print(f"LSTM model saved to {model_path}")


    # Reshape Predictions to 2D
    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)

    # Inverse Transform Predictions to Original Scale
    train_predict = inverse_transform_with_regressors(
        train_predict, X_train, scaler, features
    )
    test_predict = inverse_transform_with_regressors(
        test_predict, X_test, scaler, features
    )

    # Plotting actual vs. predicted values
    y_actual = test[target][
        seq_length:
    ]  # Skip initial seq_length for which we don't have predictions
    y_pred = test_predict

    plt.figure(figsize=(12, 6))
    plt.plot(
        test["Date_Time"][seq_length:],
        y_actual,
        label="Actual",
        marker="o",
        linestyle="-",
    )
    plt.plot(
        test["Date_Time"][seq_length:],
        y_pred,
        label="Predicted (LSTM)",
        marker="x",
        linestyle="-",
    )
    plt.xlabel("Time")
    plt.ylabel(target)
    plt.legend()
    plt.title(f"LSTM Prediction vs. Actual for {target}")
    plt.show()

    # Error Metrics Calculation
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    # Descriptive Statistics
    std_dev = np.std(y_actual)
    mean = np.mean(y_actual)

    # Print Metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Mean: {mean:.4f}")

    return mse, rmse, mae, std_dev, mean


def inverse_transform_with_regressors(predictions, input_data, scaler, features):
    """
    Inverse transforms predictions from scaled values back to original scale, incorporating regressor variables.

    Args:
    - predictions (numpy.ndarray): Predictions in scaled format.
    - input_data (numpy.ndarray): Input data for the transformation.
    - scaler (sklearn.preprocessing.MinMaxScaler): Scaler object used for normalization.
    - features (list): List of feature names, including the target and regressors.

    Returns:
    - numpy.ndarray: Inverse transformed predictions in the original scale.
    """
    inv_predictions = []
    for i in range(len(predictions)):
        # Create a temporary array with the same shape as the original training data
        to_inverse = np.zeros((1, len(features)))
        to_inverse[0, 0] = predictions[i, 0]  # Fill in the prediction
        to_inverse[0, 1:] = input_data[i, -1, 1:]  # Fill in the regressors
        # Inverse transform and extract the target column
        inv_predictions.append(scaler.inverse_transform(to_inverse)[0, 0])
    return np.array(inv_predictions)


def random_forest_prediction(
    train, test, target, lag_features, num_estimators, rand_state, regressors=[]
):
    """
    Performs time series forecasting using a Random Forest model.

    Args:
    - train (DataFrame): Training dataset containing target and regressors.
    - test (DataFrame): Test dataset containing target and regressors.
    - target (str): Name of the target variable to forecast.
    - lag_features (int): Number of lagged features to create.
    - n_estimators (int): Number of trees in the Random Forest.
    - random_state (int): Random seed for reproducibility.
    - regressors (list, optional): List of additional regressor variables. Defaults to an empty list.

    Returns:
    - tuple: Mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE),
             standard deviation of actual values, and mean of actual values.
    """
    # Select features and create lagged features
    features = [target] + regressors
    train_data = train[features]
    test_data = test[features]

    for i in range(1, lag_features + 1):
        train_data[f"{target}_lag_{i}"] = train_data[target].shift(i)
        test_data[f"{target}_lag_{i}"] = test_data[target].shift(i)

    # Drop Rows with NaN values due to shifting
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Separate Features (X) and Target (y)
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]

    # Build Random Forest Model
    model = RandomForestRegressor(n_estimators=num_estimators, random_state=rand_state)

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
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Mean: {mean:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test["Date_Time"], y_test, label="Actual", marker="o", linestyle="-")
    plt.plot(
        test["Date_Time"],
        y_pred,
        label="Predicted (Random Forest)",
        marker="x",
        linestyle="-",
    )
    plt.xlabel("Time")
    plt.ylabel(target)
    plt.legend()
    plt.title(f"Random Forest Prediction vs. Actual for {target}")
    plt.show()

    # Return errors
    return mse, rmse, mae, std_dev, mean


def compare_models(
    train,
    test,
    targets,
    regressors_dict,
    seq_length,
    lag_features,
    random_state,
    num_estimators,
):
    """
    Compares the performance of Prophet, LSTM, and Random Forest models for multiple target variables.

    Args:
    - train (DataFrame): Training dataset.
    - test (DataFrame): Test dataset.
    - targets (list): List of target variable names to forecast.
    - regressors_dict (dict): Dictionary where keys are target variable names and values are lists of regressor variables.
    - seq_length (int): Length of sequences used for LSTM training.
    - lag_features (int): Number of lagged features for Random Forest.
    - random_state (int): Random seed for reproducibility in Random Forest.
    - num_estimators (int): Number of trees in the Random Forest.

    Returns:
    - DataFrame: Comparison table of error metrics (MSE, RMSE, MAE, STD_DEV, MEAN) for each model and target.
    """
    results = {}

    for target in targets:
        regressors = regressors_dict.get(target, [])

        # Prophet
        mse, rmse, mae, std_dev, mean = prophet_prediction(
            train, test, target, regressors
        )
        results[f"Prophet_{target}"] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "STD_DEV": std_dev,
            "MEAN": mean,
        }

        # LSTM
        mse, rmse, mae, std_dev, mean = lstm_prediction(train, test, target=target, seq_length=seq_length, regressors=regressors, save_model=True)
        results[f"LSTM_{target}"] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "STD_DEV": std_dev,
            "MEAN": mean,
        }

        # Random Forest
        mse, rmse, mae, std_dev, mean = random_forest_prediction(
            train, test, target, lag_features, num_estimators, random_state, regressors
        )
        results[f"RandomForest_{target}"] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "STD_DEV": std_dev,
            "MEAN": mean,
        }

    # Create and Display Comparison Table
    results_df = pd.DataFrame(results).T
    print("\nError Metrics Comparison:")
    print(results_df)
    return results_df
