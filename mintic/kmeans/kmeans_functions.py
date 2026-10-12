import numpy as np

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError('Points must have the same dimension')

    return np.sqrt(np.sum((point1-point2)**2))

def create_kmeans(data, n_clusters=8, max_iter=1000, tol=0.0001, randomstate=None):
    # centroid kmeans

    # in case randomstate is specified, set a random seed
    if randomstate is not None:
        np.random.seed(randomstate)

    n_samples = data.shape[0]

    # randomly select initial centroids
    random_indexes = np.random.choice(n_samples, size=n_clusters, replace=False)
    centroids = data[random_indexes].copy()

    for iteration in range(max_iter):
        # assign labels to each point based on the closest centroid
        labels = np.zeros(n_samples, dtype=int)

        # calculate distances from each point to each centroid and assign the label of the closest centroid
        for i in range(n_samples):
            distances = np.array([euclidean_distance(data[i], centroid) for centroid in centroids])
            labels[i] = np.argmin(distances)

        # calculate new centroids (mean of their assigned points)
        new_centroids = np.zeros_like(centroids)

        for j in range(n_clusters):
            points_in_cluster = data[labels == j]

            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[j] = centroids[j]  # keep old centroid if no points are assigned

        # convergence
        condition = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids

        if condition < tol:
            break

    return centroids, labels
