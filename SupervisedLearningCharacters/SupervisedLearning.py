import numpy as np
import pickle
import os
import gzip
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, num_inputs, num_hiddenLayers, num_hiddenLayer, num_outputs, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.num_hiddenLayers = num_hiddenLayers
        self.num_outputs = num_outputs
        self.layers_weights = []
        self.layers_biases = []
        self.file = "data.pkl"
        load = os.path.exists(self.file)

        if load:
            with open(self.file, "rb") as f:
                data = pickle.load(f)
                self.layers_weights = data["weights"]
                self.layers_biases = data["biases"]
        else:
            # He initialization for weights
            w = np.random.randn(num_hiddenLayer, num_inputs) * np.sqrt(2 / num_inputs)
            self.layers_weights.append(w)
            for i in range(num_hiddenLayers - 1):
                w = np.random.randn(num_hiddenLayer, num_hiddenLayer) * np.sqrt(2 / num_hiddenLayer)
                self.layers_weights.append(w)
            w = np.random.randn(num_outputs, num_hiddenLayer) * np.sqrt(2 / num_hiddenLayer)
            self.layers_weights.append(w)

        # Initialize biases
        for i in range(num_hiddenLayers):
            b = np.zeros((num_hiddenLayer, 1))
            self.layers_biases.append(b)
        b = np.zeros((num_outputs, 1))
        self.layers_biases.append(b)

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_dr(self, x):
        return x > 0

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # for numerical stability
        return exp_x / (np.sum(exp_x, axis=0) + 1e-8)  # Add small epsilon to avoid division by zero

    def forward(self, inputs):
        self.current_layers_outputs = []
        self.activated_outputs = []
        # First hidden layer
        Z = np.dot(self.layers_weights[0], inputs) + self.layers_biases[0]
        A = self.ReLU(Z)
        self.current_layers_outputs.append(Z)
        self.activated_outputs.append(A)
        # All hidden layers
        for i in range(1, len(self.layers_weights) - 1):
            Z = np.dot(self.layers_weights[i], A) + self.layers_biases[i]
            A = self.ReLU(Z)
            self.current_layers_outputs.append(Z)
            self.activated_outputs.append(A)
        # Output layer
        Z = np.dot(self.layers_weights[-1], A) + self.layers_biases[-1]
        A = self.softmax(Z)
        return A

    def one_hot(self, Y):
        one_hot_Y = np.zeros((self.num_outputs, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    def backward(self, inputs, expected_outputs, predicted_outputs):
        m = expected_outputs.size
        one_hot_y = self.one_hot(expected_outputs)

        # Output layer
        dZ = predicted_outputs - one_hot_y
        dW = (1 / m) * dZ.dot(self.activated_outputs[-1].T)
        dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # Gradient clipping
        dW = np.clip(dW, -1, 1)
        dB = np.clip(dB, -1, 1)
        # Update output weights and biases
        self.layers_weights[-1] -= self.learning_rate * dW
        self.layers_biases[-1] -= self.learning_rate * dB

        # Hidden layers
        for i in range(self.num_hiddenLayers - 1, 0, -1):
            dZ = self.layers_weights[i + 1].T.dot(dZ) * self.ReLU_dr(self.current_layers_outputs[i])
            dW = (1 / m) * dZ.dot(self.activated_outputs[i - 1].T)
            dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            # Gradient clipping
            dW = np.clip(dW, -1, 1)
            dB = np.clip(dB, -1, 1)
            # Update weights and biases
            self.layers_weights[i] -= self.learning_rate * dW
            self.layers_biases[i] -= self.learning_rate * dB

        # First hidden layer
        dZ = self.layers_weights[1].T.dot(dZ) * self.ReLU_dr(self.current_layers_outputs[0])
        dW = (1 / m) * dZ.dot(inputs.T)
        dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # Gradient clipping
        dW = np.clip(dW, -1, 1)
        dB = np.clip(dB, -1, 1)
        # Update weights and biases
        self.layers_weights[0] -= self.learning_rate * dW
        self.layers_biases[0] -= self.learning_rate * dB

    def cross_entropy_loss(self, predicted_outputs, expected_outputs):
        m = expected_outputs.size
        epsilon = 1e-10  # Small constant to avoid log(0)
        predicted_outputs = np.clip(predicted_outputs, epsilon, 1 - epsilon)  # Clip the predictions
        log_probs = -np.log(predicted_outputs[expected_outputs, np.arange(m)])
        loss = np.sum(log_probs) / m
        return loss

    def train(self, X, y, epochs):
        X = X.T
        y = y.reshape(1, -1)
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 10 == 0:
                prediction = np.argmax(output, axis=0)
                acc = np.mean(y.flatten() == prediction)
                loss = self.cross_entropy_loss(output, y)
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {acc * 100}%")

                # Save weights and biases
                with open(self.file, 'wb') as f:
                    pickle.dump({"weights": self.layers_weights, "biases": self.layers_biases}, f)
