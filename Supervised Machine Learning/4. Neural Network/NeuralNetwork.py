import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """
        Initialize the neural network with a list of layers.

        Parameters:
            layers (list): List of layers, where each layer is an instance of a neuron class.
        """
        self.layers = layers

    def forward(self, X):
        """
        Perform forward propagation through all layers.

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).

        Returns:
            np.array: Output of the final layer.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true):
        """
        Perform backward propagation to compute gradients for each layer.

        Parameters:
            y_true (np.array): True labels of shape (n_samples,).
        """
        # Start with the gradient of the loss with respect to the output
        dA = self.layers[-1].compute_loss_gradient(y_true, self.activations[-1])

        # Iterate through layers in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = self.activations[i]  # Input to the current layer

            # Compute gradients for the current layer
            dW, db, dA_prev = layer.backward(A_prev, dA)

            # Update weights and bias
            layer.weights -= self.learning_rate * dW
            layer.bias -= self.learning_rate * db

            # Pass the gradient to the previous layer
            dA = dA_prev

    def train(self, X, y, learning_rate=0.01, n_iters=1000):
        """
        Train the neural network using gradient descent.

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).
            y (np.array): True labels of shape (n_samples,).
            learning_rate (float): Learning rate for gradient descent.
            n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate

        for _ in range(n_iters):
            # Forward propagation
            output = self.forward(X)

            # Compute loss
            loss = self.layers[-1].compute_loss(y, output)

            # Backward propagation
            self.backward(y)

            # Print loss every 100 iterations
            if _ % 100 == 0:
                print(f"Iteration {_}, Loss: {loss}")


    def predict(self, X, threshold=0.5):
        """
        Make predictions using the neural network.

        Parameters:
            X (np.array): Input data of shape (n_samples, n_features).
            threshold (float): Threshold for binary classification.

        Returns:
            np.array: Binary predictions of shape (n_samples,).
        """
        output = self.forward(X)
        return (output >= threshold).astype(int)