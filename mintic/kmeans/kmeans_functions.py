import numpy as np

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError('Points must have the same dimension')

    return np.sqrt(np.sum((point1-point2)**2))