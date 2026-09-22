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

def predict_tree(tree, x):
    node = tree # root

    while isinstance(node, dict): # while node is not leaf
        value = x[node['feature']] # vakue of the feature

        if value in node['children']: # if value is in the children of the node, go to the children
            node = node['children'][value]

        else: # if not, default child (most common class)
            node = node['default']

    return node

# --------------------------------------------------------------------


def build_id3_tree(X_sample, y_sample):
    # since this algorithm is recursive and long it's convenient to separate in base cases and recursive step
    # the base cases are, first, when all the labels are the same
    # second, when there are no features left to split on (return the most common class)

    if len(np.unique(y_sample)) == 1:
        return y_sample[0] # return that label

    # case 2

    if X_sample.shape[1] == 0:
        classes, counts = np.unique(y_sample, return_counts=True)
        return classes[np.argmax(counts)] # most common class

    # now the recursive step
    # which feature gives the highest info gain?

    highest_gain = -1
    best_feature = 0

    for col in range(X_sample.shape[1]):
        info_gain = information_gain(X_sample[:, col], y_sample) # info gain of the feature column

        if info_gain > highest_gain: # if info gain is higher than the previous, the highest gain is updated along the best feature
            highest_gain = info_gain
            best_feature = col

    # build branch

    feature_values = np.unique(X_sample[:, best_feature]) # unique values of the feature column
    children = {} # store children of the node

    for value in feature_values:
        mask = X_sample[:, best_feature] == value # mask for rows with current value

        X_subset = X_sample[mask] # subset of the features
        y_subset = y_sample[mask] # subset of the labels

        children[value] = build_id3_tree(np.delete(X_subset, best_feature, axis=1), y_subset) # recursion, it's important to delete the feature column since we don't wanna use it again

    return {'feature': best_feature, 'children': children, 'default': np.bincount(y_sample).argmax()}

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