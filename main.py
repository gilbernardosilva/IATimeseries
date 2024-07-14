import pandas as pd
import matplotlib.pyplot as plt
from utils import regressor_co2, regressor_particles
from functions import clean_data, extract_datetime, data_analysis, divide_data, compare_models
DATASET_RATIO = 0.3

# Load the Dataset
dataset2 = pd.read_csv('dataset_210.csv')
print(dataset2.shape)       # Display the number of rows and columns
print(dataset2.head())      # Show the first 5 rows for a quick look

# Drop Unnecessary Columns
dataset2 = dataset2.drop('ClassId', axis=1)  # Remove 'ClassId' column
dataset2 = dataset2.drop('classroom', axis=1) # Remove 'classroom' column

# Clean the Data (Custom Function)
dataset2 = clean_data(dataset2)  # Apply your data cleaning logic

# Data Inspection After Cleaning
print(dataset2.shape)      # Display the updated shape after cleaning
print(dataset2.info())     # Summary of data types and non-null values
print(dataset2.describe()) # Statistical overview (for numeric columns)


dataset2.boxplot(column=["Particules 1"])
plt.title('Distribution of Particules 1')
plt.ylabel('Particules 1 Concentration (ppm)')
plt.show()

dataset2.boxplot(column=["Particules 2.5"])
plt.title('Distribution of Particules 2.5')
plt.ylabel('Particules 2.5 Concentration (ppm)')
plt.show()

dataset2.boxplot(column=["Particules 10"])
plt.title('Distribution of Particules 10')
plt.ylabel('Particules 10 Concentration (ppm)')
plt.show()

dataset2.boxplot(column=["CO2"])
plt.title('Distribution of CO2')
plt.ylabel('CO2 Concentration (ppm)')
plt.show()

dataset = extract_datetime(dataset2) # Converts Time column to Date and Time (hours)
print(dataset['Date_Time'].dtype)

data_analysis(dataset,['Particules 1', 'Particules 2.5', 'Particules 10', 'CO2']) #Graphical representation of the columns over time

train,test = divide_data(DATASET_RATIO,dataset) #Divides the dataset into two for testing and training using a ratio

targets = ['Particules 1', 'Particules 2.5', 'Particules 10', 'CO2']
regressors_dict = {'Particules 1': regressor_particles, 'Particules 2.5': regressor_particles, 
                   'Particules 10': regressor_particles, 'CO2': regressor_co2}

results_df = compare_models(train, test, targets, regressors_dict, seq_length=10, lag_features=10)