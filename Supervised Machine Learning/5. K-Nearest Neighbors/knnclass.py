import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Defining the Euclidean distance function - Will Be Default
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Defining Manhattan_Distance

def manhattan_distance(point1, point2):
    return np.abs(point1[0]-point2[0]) + np.abs(point1[1]-point2[1])

# Accuracy Function for Evaluating Classification Accuracy

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Step 2: Implement the KNN class
class KNN:
    def __init__(self, k=3, distance_function = euclidean_distance, type = 1):
        self.k = k  # Number of neighbors
        self.distance = distance_function # Default is using euclidean distance
        self.k = type # 1 for classification, 0 for regression

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute the distance between x and all points in the training data
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels/values of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.type == 1:

            # Return the most common class label among the k neighbors
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        
        else:

            # Return the mean of the k_nearest_values
            average_value = np.mean(k_nearest_labels)
            return average_value
