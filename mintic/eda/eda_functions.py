import pandas as pd
import matplotlib.pyplot as plt

def impute_missing(data, strategy, columns=None):
    imputed_data = data.copy()

    # this is so the columns parameter can be optional, it will just select all columns of the df
    if columns is None:
        columns = imputed_data.columns

    for col in columns:
        non_null_values = imputed_data[col].dropna()

        if strategy == 'mean':
            imputed_values = non_null_values.sum() / len(non_null_values)

        elif strategy == 'median':
            sorted_values = non_null_values.sort_values()
            n = len(sorted_values)
            # if the length of the sorted values is even, the median is the average of the two middle valuess
            # if the length of the sorted values is odd, the median is the middle value as it's the only value in the middle
            if n % 2 == 0:
                mid_index = n // 2
                imputed_values = (sorted_values.iloc[mid_index - 1] + sorted_values.iloc[mid_index]) / 2
            else:
                imputed_values = sorted_values.iloc[n // 2]

        elif strategy == 'mode':
            # the mode is just the most frequent value in the column, so value_counts can be used to identify it
            frequency = non_null_values.value_counts()
            imputed_values = frequency.index[0]

        # lastly, we just fill the missing values with whatever strategy was chosen, with inplace to modify the df and not make a copy
        imputed_data[col].fillna(imputed_values, inplace=True)

    return imputed_data

def detect_outliers(data, method, threshold):
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (IQR*threshold)
        upper_bound = Q3 + (IQR*threshold)

        # in the iqr method the outliers are the values that are below the lower bound or above the upper bound (below quantile 25 or above quantile 75) 
        # this is the variable that detects those values
        outliers = data[(data < lower_bound) | (data > upper_bound)]

    elif method == 'z_score':
        non_null_data = data.dropna()
        mean = non_null_data.sum() / len(non_null_data)
        std = (((non_null_data - mean)**2).sum() / len(non_null_data)) ** 0.5

        # in the z score method the outliers are the values that are above or below a threshold of std from the mean
        z_scores = (data - mean) / std
        outliers = data[abs(z_scores) > threshold]

    return outliers

def handle_outliers(data, method, action, threshold):
    handled_data = data.copy()

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (IQR*threshold)
        upper_bound = Q3 + (IQR*threshold)

        # when handling outliers, we can use either trim or cap
        # trim: remove the outliers from the df within the bounds
        if action == 'trim':
            handled_data = handled_data[(handled_data >= lower_bound) & (handled_data <= upper_bound)]

        # cap: replace the outliers with the bounds threshold values
        elif action == 'cap':
            handled_data[handled_data < lower_bound] = lower_bound
            handled_data[handled_data > upper_bound] = upper_bound

    elif method == 'zscore':
        non_null_data = handled_data.dropna()
        mean = non_null_data.sum() / len(non_null_data)
        std = (((non_null_data - mean)**2).sum() / len(non_null_data)) ** 0.5

        z_scores = (handled_data - mean) / std

        # similar to the iqr method, we can either trim or cap the outliers based on the z score threshold
        # we also used absolute value of the z score here, similar to what we did in detect_outliers.py to handle both positive and negative outliers without having to do it separately
        if action == 'trim':
            handled_data = handled_data[abs(z_scores) <= threshold]
        elif action == 'cap':
            lower_bound = mean - (std*threshold)
            upper_bound = mean + (std*threshold)
            handled_data[handled_data < lower_bound] = lower_bound
            handled_data[handled_data > upper_bound] = upper_bound

    return handled_data

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
