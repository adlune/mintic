import pandas as pd

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
            
