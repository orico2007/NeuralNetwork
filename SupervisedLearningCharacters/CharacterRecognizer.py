import numpy as np
import os
import gzip
import pickle
from matplotlib import pyplot as plt
import SupervisedLearning

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

# Create a neural network instance
nn = SupervisedLearning.NeuralNetwork(784, 2, 128, 62, 0.001)  # Use a lower learning rate

# Train the model
nn.train(X_train, Y_train, epochs=1000)