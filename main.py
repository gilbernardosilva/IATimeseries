import pandas as pd
import matplotlib.pyplot as plt
from functions import clean_data, extract_datetime, DataAnalyis, forecast_with_neuralprophet


# 1. Load the Dataset
dataset2 = pd.read_csv('dataset_106.csv')
print(dataset2.shape)       # Display the number of rows and columns
print(dataset2.head())      # Show the first 5 rows for a quick look

# 2. Drop Unnecessary Columns
dataset2 = dataset2.drop('ClassId', axis=1)  # Remove 'ClassId' column
dataset2 = dataset2.drop('classroom', axis=1) # Remove 'classroom' column

# 3. Clean the Data (Custom Function)
dataset2 = clean_data(dataset2)  # Apply your data cleaning logic

# 4. Data Inspection After Cleaning
print(dataset2.shape)      # Display the updated shape after cleaning
print(dataset2.info())     # Summary of data types and non-null values
print(dataset2.describe()) # Statistical overview (for numeric columns)

# Particules 2.5,Particules 10,CO2
dataset2.boxplot(column=["Particules 1"])

# Set plot title and axis labels
plt.title('Distribution of Particules 1')
plt.ylabel('Particules 1 Concentration (ppm)')

# Show the plot
#plt.show()


dataset2.boxplot(column=["Particules 2.5"])

# Set plot title and axis labels
plt.title('Distribution of Particules 2.5')
plt.ylabel('Particules 2.5 Concentration (ppm)')

# Show the plot
#plt.show()

dataset2.boxplot(column=["Particules 10"])

# Set plot title and axis labels
plt.title('Distribution of Particules 10')
plt.ylabel('Particules 10 Concentration (ppm)')

# Show the plot
#plt.show()


dataset2.boxplot(column=["CO2"])

# Set plot title and axis labels
plt.title('Distribution of CO2')
plt.ylabel('CO2 Concentration (ppm)')


# Show the plot
#plt.show()


dataset = extract_datetime(dataset2)

DataAnalyis(dataset,['Particules 1', 'Particules 2.5', 'Particules 10', 'CO2'])
print(dataset['Date_Time'])


Particules1Result = forecast_with_neuralprophet(dataset, 'Particules 1')
#Particules25Result = forecast_with_neuralprophet(dataset, 'Particules 2.5')
#Particules10Result = forecast_with_neuralprophet(dataset, 'Particules 10')
#CO2Result = forecast_with_neuralprophet(dataset, 'CO2')

