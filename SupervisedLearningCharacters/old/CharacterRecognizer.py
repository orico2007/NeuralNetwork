import numpy as np
import os
import gzip
import pickle
from matplotlib import pyplot as plt

def load_idx_data(file_name):
    with gzip.open(file_name, 'rb') as f:
        f.read(4)
        num_items = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows * cols)
    return data

def load_labels(file_name):
    with gzip.open(file_name, 'rb') as f:
        f.read(4)
        num_items = int.from_bytes(f.read(4), byteorder='big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

train_images_file = 'emnist-letters-train-images-idx3-ubyte.gz'
train_labels_file = 'emnist-letters-train-labels-idx1-ubyte.gz'
test_images_file = 'emnist-letters-test-images-idx3-ubyte.gz'
test_labels_file = 'emnist-letters-test-labels-idx1-ubyte.gz'

X_train = load_idx_data(train_images_file) / 255.0
Y_train = load_labels(train_labels_file)

X_test = load_idx_data(test_images_file) / 255.0
Y_test = load_labels(test_labels_file)

shuffle_indices = np.random.permutation(len(X_train))
X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]

def init_params():
    W1 = np.random.randn(128, 784) * 0.01
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(128, 128) * 0.01
    b2 = np.zeros((128, 1))
    W3 = np.random.randn(62, 128) * 0.01
    b3 = np.zeros((62, 1))
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X.T) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y, num_classes=62):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, W3, Z3, A3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def save_weights(W1, b1, W2, b2, W3, b3, filename="weights.pkl"):
    with open(filename, "wb") as file:
        pickle.dump((W1, b1, W2, b2, W3, b3), file)
    print(f"Model weights saved to {filename}")

def load_weights(filename="weights.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            W1, b1, W2, b2, W3, b3 = pickle.load(file)
        print(f"Model weights loaded from {filename}")
        return W1, b1, W2, b2, W3, b3
    else:
        print("No saved model found. Initializing new weights.")
        return init_params()

def gradient_descent(X, Y, iterations, alpha, save_interval=10, load_existing=True):
    if load_existing:
        W1, b1, W2, b2, W3, b3 = load_weights()
    else:
        W1, b1, W2, b2, W3, b3 = init_params()
    
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, W2, W3, Z3, A3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A3)
            print(f"Iteration {i}: Accuracy = {get_accuracy(predictions, Y)}")

        if i % save_interval == 0:
            save_weights(W1, b1, W2, b2, W3, b3)

    save_weights(W1, b1, W2, b2, W3, b3)

gradient_descent(X_train, Y_train, 10000, 0.05)
print("Training complete.")
