import numpy as np

def bootstrap_sample(X, y, random_state=None):
    # for reproducibility generally 42 is used, but this is so that the user can set their random state if they want
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indexes = np.random.choice(n_samples, size=n_samples, replace=True)
    
    X_sample = X[indexes]
    y_sample = y[indexes]
    
    return X_sample, y_sample


# auxiliary functions for building
# --------------------------------------------------------------------

def entropy(y):
    # entropy of a unidimensional array of labels
    # according to shannon's entropy formula:

    value, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy_value = -np.sum(probabilities * np.log2(probabilities)) # formula for entropy
    return entropy_value

def information_gain(X_column, y):
    pass

# --------------------------------------------------------------------


def build_id3_tree(X_sample, y_sample):
    pass

def build_random_forest(X, y, n_trees=10, random_state=None):
    pass

def predict_ensemble(forest, X_test):
    pass