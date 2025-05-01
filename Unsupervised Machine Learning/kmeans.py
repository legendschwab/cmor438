import numpy as np

# Defining the Euclidean distance function - Will Be Default
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Manhattan Distance 
def manhattan_distance(point1, point2):
    return np.abs(point1[0]-point2[0]) + np.abs(point1[1]-point2[1])

# Returns the coordinates of the centroid given the 2D np array cluster, which lists the coordinates of the points in a cluster
def calculate_centroid(cluster):
    centroid = np.mean(cluster, axis=0)
    return centroid

# Given data and centroids, assigns each point to an array in the output clusters
def assign_clusters(data, centroids, distance):
    clusters = [[] for i in centroids] # Initialize empty list of lists for clusters
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        closest_index = np.argmin(distances)
        clusters[closest_index].append(point)
    return clusters

# Returns the coordinates of the new centroids
def update_centroids(data, clusters):
    new_centroids = []
    for cluster in clusters:
        # Case where no points are assigned to a centroid point
        if len(cluster) == 0:
            random_index = np.random.choice(len(data),1)
            new_centroid = data[random_index][0]
        else:
            new_centroid = calculate_centroid(np.array(cluster))
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Checks if newly calculated centroids have changed within a tolerance level 
def has_converged(old_centroids, new_centroids, tol=1e-4):
    """
    Check for convergence in k-means cluster

    Inputs: old_centroids: numpy array with coordinates for old_centroids
            new_centroids: numpy array with coordinates for new_centroids
            tol: tolerance for convergence
    Output: True if difference between new and old centroids are all within the tolerance
    """
    # Compute the distance between corresponding centroids
    centroid_diff = np.linalg.norm(old_centroids - new_centroids, axis=1)
    
    # Check if all the distances are below the tolerance
    return np.all(centroid_diff < tol)

# Puts all the helper functions together and implements k-means clustering

def k_means(data, k, distance_function, max_iters = 100):
    """
    Performs k-means clustering 

    Inputs: data: numpy array where each row is a data point
            k: The number of clusters
            max_iters: Max number of iterations of algorithm, default is 100
            distance: the distance function used
    Output: centroids: numpy array where each row is a centroid of a cluster
            clusters: a list of lists that gives the coordinates of each point in the cluster
    """

    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]

    for iter in range(max_iters):
        # Assign clusters and update centroids
        clusters = assign_clusters(data, centroids, distance_function)
        new_centroids = update_centroids(data, clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters

# Computes the Within-Cluster Sum of Squares

def compute_wcss(clusters, centroids, distance_function):
    """
    Computes the Within-Cluster Sum of Squares (WCSS)
    
    Inputs:
        clusters: List of clusters, each containing the points in that cluster.
        centroids: numpy array that gives centroids of the clusters 
    
    Output:
        The WCSS value.
    """
    wcss = 0
    k = len(clusters)  # Number of clusters
    
    for i in range(k):
        cluster_points = clusters[i]  # Points in the i-th cluster
        centroid = centroids[i]  # Centroid of the i-th cluster
        
        # Calculate the sum of squared distances from each point in the cluster to the centroid
        cluster_wcss = np.sum([distance_function(point, centroid) ** 2 for point in cluster_points])
        wcss += cluster_wcss
    
    return wcss

