
import numpy as np

def clean_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("numero de linhas antes de tirar as linhas com valor null ", len(data))
    data.dropna(inplace=True)
    print("numero de linhas antes de tirar as linhas com valores  duplicados ", len(data))
    data.drop_duplicates(inplace=True)
    print("o numero de linhas final é ", len(data))
    return data

