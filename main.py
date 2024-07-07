import pandas as pd
from functions import clean_data


dataset2 = pd.read_csv('dataset_106.csv')
print(dataset2.shape)
print(dataset2.head())

dataset2 = dataset2.drop('ClassId',axis=1)
dataset2 = dataset2.drop('classroom',axis=1)
dataset2 = clean_data(dataset2)


print(dataset2.shape)

print(dataset2.info())

print(dataset2.describe())
