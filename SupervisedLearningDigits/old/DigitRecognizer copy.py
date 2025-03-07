import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle
from PIL import Image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

shuffle_indices = np.random.permutation(len(X_train))
X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]

def init_prams():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_prams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def save_weights(W1, b1, W2, b2, filename="weights.pkl"):
    with open(filename, "wb") as file:
        pickle.dump((W1, b1, W2, b2), file)
    print(f"Model weights saved to {filename}")

def load_weights(filename="weights.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            W1, b1, W2, b2 = pickle.load(file)
        print(f"Model weights loaded from {filename}")
        return W1, b1, W2, b2
    else:
        print("No saved model found. Initializing new weights.")
        return init_prams()







"""-----------------------Training------------------------"""

def gradient_descent(X, Y, iterations, alpha, save_interval=10, load_existing=True):
    if load_existing:
        W1, b1, W2, b2 = load_weights()
    else:
        W1, b1, W2, b2 = init_prams()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_prams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(f"Iteration {i}: Accuracy = {get_accuracy(predictions, Y):.4f}")

        if i % save_interval == 0:
            save_weights(W1, b1, W2, b2)

    save_weights(W1, b1, W2, b2)
    return W1, b1, W2, b2

#W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 10000, 0.1)

#print("Training complete.")







"""-----------------------Testing Random Image------------------------"""

def make_predictions(X,W1,b1,W2,b2):
    Z1, A1, Z2, A2 = forward_prop(W1,b1,W2,b2,X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index):
    W1, b1, W2, b2 = load_weights()
    
    current_image = X_train[index, :].reshape(-1, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    
    print("Prediction:", prediction)
    print("Label:", label)

    current_image = current_image.reshape(28, 28) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()

#test_prediction(654)




"""-----------------------Testing Image Upload------------------------"""


def predict_image(image_path):
    W1, b1, W2, b2 = load_weights()
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))

    image = np.array(image).reshape(-1, 1) / 255.0
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, image)

    predicted_class = np.argmax(A2, axis=0)

    confidence = A2[predicted_class, 0] * 100

    print(f"Confidence: {confidence}")
    print(f"Prediction: {predicted_class}")

    plt.imshow(image.reshape(28, 28) * 255, cmap='gray')
    plt.show()

    return predicted_class, confidence

path = "image5.png"
prediction, confidence = predict_image(path)