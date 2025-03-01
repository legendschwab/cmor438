import numpy as np

class LogNeuron:
    def __init__(self, n_features):
        """
        Initialize the neuron with random weights and zero bias.

        Parameters:
            n_features (int): Number of input features.
        """
        self.weights = np.random.randn(n_features)  # Initialize weights randomly
        self.bias = 0  # Initialize bias to zero

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        Parameters:
            z (float or np.array): Input to the sigmoid function.

        Returns:
            float or np.array: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Perform forward propagation (compute predictions).

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).

        Returns:
            np.array: Predicted probabilities of shape (n_samples,).
        """
        z = np.dot(X, self.weights) + self.bias  # Compute weighted sum
        return self.sigmoid(z)  # Apply sigmoid function

    def compute_loss(self, y_true, y_pred):
        """
        Compute the binary cross-entropy loss.

        Parameters:
            y_true (np.array): True labels of shape (n_samples,).
            y_pred (np.array): Predicted probabilities of shape (n_samples,).

        Returns:
            float: Binary cross-entropy loss.
        """
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute gradients for weights and bias.

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).
            y_true (np.array): True labels of shape (n_samples,).
            y_pred (np.array): Predicted probabilities of shape (n_samples,).

        Returns:
            tuple: Gradients for weights and bias.
        """
        error = y_pred - y_true  # Compute error
        dw = np.dot(X.T, error) / len(y_true)  # Gradient for weights
        db = np.mean(error)  # Gradient for bias
        return dw, db

    def train(self, X, y, learning_rate=0.01, n_iters=1000):
        """
        Train the neuron using gradient descent.

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).
            y (np.array): True labels of shape (n_samples,).
            learning_rate (float): Learning rate for gradient descent.
            n_iters (int): Number of iterations for training.
        """
        for _ in range(n_iters):
            # Forward propagation
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, y_pred)

            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)

            # Update weights and bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Print loss every 100 iterations
            #if _ % 100 == 0:
            #    print(f"Iteration {_}, Loss: {loss}")

    def compute_loss_gradient(self, y_true, y_pred):
        """
        Compute the gradient of the loss with respect to the output.

        Parameters:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted probabilities.

        Returns:
            np.array: Gradient of the loss with respect to the output.
        """
        return (y_pred - y_true) / len(y_true)

    def backward(self, A_prev, dA):
        """
        Compute gradients for the logistic regression layer.

        Parameters:
            A_prev (np.array): Input to the layer (output of the previous layer).
            dA (np.array): Gradient of the loss with respect to the output of this layer.

        Returns:
            tuple: Gradients for weights, bias, and input.
        """
        dZ = dA  # For the output layer, dZ = dA
        dW = np.dot(A_prev.T, dZ) / len(A_prev)
        db = np.mean(dZ, axis=0)
        dA_prev = np.dot(dZ, self.weights.T)
        return dW, db, dA_prev

    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using a threshold.

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).
            threshold (float): Threshold for binary classification.

        Returns:
            np.array: Binary predictions of shape (n_samples,).
        """
        y_pred = self.forward(X)  # Compute predicted probabilities
        return (y_pred >= threshold).astype(int)  # Convert probabilities to binary predictions