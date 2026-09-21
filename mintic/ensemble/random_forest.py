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
    H_Y = entropy(y) # this is the entropy of the labels

    value, counts = np.unique(X_column, return_counts=True) # for feature column
    probabilities = counts / len(X_column)

    # conditional entropy H(Y|X)
    H_Y_X = 0
    for v, p in zip(value, probabilities):
        y_subset = y[X_column == v]
        H_Y_X += p * entropy(y_subset)

    # information gain is just the difference between the entropy of the labels and the conditional entropy
    information_gain = H_Y - H_Y_X
    return information_gain

def predict_tree():
    pass

# --------------------------------------------------------------------


def build_id3_tree(X_sample, y_sample):
    pass

def build_random_forest(X, y, n_trees=10, random_state=None):
    forest = [] # to store the trees of the forest

    for i in range(n_trees):
        seed = (random_state + i) if random_state is not None else None # this is so each tree has a different random state (incremented by 1 each iteration). it's important since if we didn't have this all trees would be the same
        X_sample, y_sample = bootstrap_sample(X, y, random_state=seed)
        tree = build_id3_tree(X_sample, y_sample)
        forest.append(tree)

    return forest

def predict_ensemble(forest, X_test):
    predictions = [] # to store the predictions of each tree

    for x in X_test:
        tree_predictions = [predict_tree(tree, x) for tree in forest]

        # majority voting
        classes, counts = np.unique(tree_predictions, return_counts=True)
        candidates = classes[counts == np.max(counts)] # in case of a tie
        predictions.append(candidates[0]) # select the first candidate

    return np.array(predictions)