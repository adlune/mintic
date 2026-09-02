import pandas as pd
import matplotlib.pyplot as plt

def plot_missing(data):
    # first we have to count the missing values in total for each column
    missing_count = data.isnull().sum()
    # this is so we can plot only the columns that have missing values, as otherwise the plot will try to plot all the columns even if they don't have any missing values
    missing_count = missing_count[missing_count > 0]

    plt.figure(figsize=(10, 5))
    plt.bar(missing_count.index, missing_count.values)
    plt.title('Cantidad de valores faltantes en el DF')
    plt.xlabel('Columnas')
    plt.ylabel('Cantidad de valores faltantes')
    plt.tight_layout()
    plt.show()
