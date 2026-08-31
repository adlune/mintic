import pandas as pd

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