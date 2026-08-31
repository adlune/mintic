import pandas as pd

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
