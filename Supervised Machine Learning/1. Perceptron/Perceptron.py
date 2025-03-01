import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure labels are either -1 or 1
        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                # Update weights and bias
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
    
    def backward(self, A_prev, dA):
        """
        Compute gradients for the perceptron layer.

        Parameters:
            A_prev (np.array): Input to the layer (output of the previous layer).
            dA (np.array): Gradient of the loss with respect to the output of this layer.

        Returns:
            tuple: Gradients for weights, bias, and input.
        """
        dZ = dA  # For a linear layer, dZ = dA
        dW = np.dot(A_prev.T, dZ) / len(A_prev)
        db = np.mean(dZ, axis=0)
        dA_prev = np.dot(dZ, self.weights.T)
        return dW, db, dA_prev

# Example usage:
if __name__ == "__main__":
    # Example dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])

    # Create and train the perceptron
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X, y)

    # Make predictions
    predictions = p.predict(X)
    print("Predictions:", predictions)