K-Nearest Neighbors is a classification algorithm that categorizes data points based on the categories of the k "closest" training data points. This idea of "closest" usually refers to L2 Euclidean distance. 

Given the data set, we can split it into training and training. For each training data set, we find the k closest training data points based on feature values. Out of the k, we see which category label is most common and assign it to this training data set. 

How do you determine an appropriate value of k? Often times, an odd k value is used so that there will never be a tie between the top 2 categories.
The elbow method is a common method to see which k value is most optimal. By plotting classification accuracy (or some other appropriate metric) against multiple k values, we can find the "elbow point", which is where the improvement rate decreases. This k value balances between accuracy and efficiency.



