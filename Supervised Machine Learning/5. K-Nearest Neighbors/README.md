**K-Nearest Neighbors** is a **non-parametric** learning algorithm that labels data points with a category or predicts a value by inferring information from the k "closest" training data points.

We can do both classification and regression with k-nearest neighbors. For
classification, we find the k-closest data points based on feature values to our new data point. Out of the k, we see which category label is most common and assign it to this new data point. If instead, we take the average label value from the neighbors, we get a regression problem.

How do you determine an appropriate value of k? Often times, an odd k value is used so that there will never be a tie between the top 2 categories.
The elbow method is a common method to see which k value is most optimal. By plotting classification accuracy (or some other appropriate metric) against multiple k values, we can find the "elbow point", which is where the rate at which accuracy improves decreases. This k value balances between accuracy and efficiency.

What does "closest" mean? This idea of "closest" usually refers to L2 **Euclidean** distance but other forms of distance can be used. We will explore both Euclidean distance and Manhattan distance when applying this algorithm.


-- Notes from Class:
-- Curse of dimensionality, high dimensional space is essentially empty, each feature vector is mapped to the majority label of the training instances, have to store all of those instances, memory issues
-- Test set should not be used to pick k. Instead, we should create a separate validation set. Test set should never influence any parameters of the model.


