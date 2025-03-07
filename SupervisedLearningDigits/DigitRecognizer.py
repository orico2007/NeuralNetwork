import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle
from PIL import Image
import SupervisedLearning

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0


shuffle_indices = np.random.permutation(len(X_train))
X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]

nn = SupervisedLearning.NeuralNetwork(784, 2, 10, 10, 0.1, True)

nn.train(X_train, Y_train, epochs=100)