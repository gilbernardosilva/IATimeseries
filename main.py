import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import constants and configurations
from utils import (
    DATASET_RATIO,
    SEQ_LENGTH,
    LAG_FEATURES,
    TARGETS,
    REGRESSORS_DICT,
    NUM_ESTIMATORS,
    RANDOM_STATE,
)

# Import custom functions for data processing and analysis
from functions import (
    clean_data,
    extract_datetime,
    data_analysis,
    divide_data,
    compare_models,
)


# Load the Dataset
dataset = pd.read_csv("dataset_210.csv")  # Read the dataset from a CSV file
print(dataset.shape)  # Display the number of rows and columns in the dataset
print(dataset.head())  # Show the first 5 rows of the dataset for a quick look

# Drop Unnecessary Columns
dataset = dataset.drop("ClassId", axis=1)  # Remove the 'ClassId' column
dataset = dataset.drop("classroom", axis=1)  # Remove the 'classroom' column

# Clean the Data (Custom Function)
dataset = clean_data(dataset)  # Apply the custom data cleaning function

# Data Inspection After Cleaning
print(dataset.shape)  # Display the updated shape of the dataset after cleaning
print(dataset.info())  # Print a summary of data types and non-null values
print(dataset.describe())  # Print statistical overview of numeric columns

# Boxplots for Data Visualization
dataset.boxplot(column=["Particules 1"])  # Create a boxplot for 'Particules 1'
plt.title("Distribution of Particules 1")  # Set the title for the boxplot
plt.ylabel("Particules 1 Concentration (ppm)")  # Set the y-axis label
plt.show()  # Display the boxplot

dataset.boxplot(column=["Particules 2.5"])  # Create a boxplot for 'Particules 2.5'
plt.title("Distribution of Particules 2.5")  # Set the title for the boxplot
plt.ylabel("Particules 2.5 Concentration (ppm)")  # Set the y-axis label
plt.show()  # Display the boxplot

dataset.boxplot(column=["Particules 10"])  # Create a boxplot for 'Particules 10'
plt.title("Distribution of Particules 10")  # Set the title for the boxplot
plt.ylabel("Particules 10 Concentration (ppm)")  # Set the y-axis label
plt.show()  # Display the boxplot

dataset.boxplot(column=["CO2"])  # Create a boxplot for 'CO2'
plt.title("Distribution of CO2")  # Set the title for the boxplot
plt.ylabel("CO2 Concentration (ppm)")  # Set the y-axis label
plt.show()  # Display the boxplot

# Extract DateTime from the Data (Custom Function)
dataset = extract_datetime(dataset)  # Convert 'Time' column to 'Date_Time' column
print(dataset["Date_Time"].dtype)  # Print the data type of the 'Date_Time' column

# Data Analysis (Custom Function)
data_analysis(dataset, ["Particules 1", "Particules 2.5", "Particules 10", "CO2"])
# Plot graphical representation of the specified columns over time


# Compute the correlation matrix for the dataset
correlation_matrix = dataset.corr()

# Set the size of the figure
plt.figure(figsize=(10, 8))

# Create a heatmap to visualize the correlation matrix
# - annot=True adds the correlation coefficient values inside the cells
# - cmap="coolwarm" sets the color scheme of the heatmap
# - fmt=".2f" formats the correlation coefficient values to 2 decimal places
# - linewidths=0.5 sets the width of the lines that will divide the cells
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Set the title of the heatmap
plt.title("Matriz de Correlação entre Variáveis")

# Display the heatmap
plt.show()

# Divide the Dataset into Training and Testing Sets (Custom Function)
train, test = divide_data(
    DATASET_RATIO, dataset
)  # Split the dataset into training and testing sets using the specified ratio

# Model Comparison (Custom Function)
results_df = compare_models(
    train,
    test,
    targets=TARGETS,
    regressors_dict=REGRESSORS_DICT,
    seq_length=SEQ_LENGTH,
    lag_features=LAG_FEATURES,
    num_estimators=NUM_ESTIMATORS,
    random_state=RANDOM_STATE,
)  # Compare different models using the training and testing sets
