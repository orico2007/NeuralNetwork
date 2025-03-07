# Neural Network for Digit and Character Recognition

This project implements an abstract class of a neural network for both digit recognition and character recognition tasks. It uses a fully connected feedforward neural network with ReLU activations and softmax output. The neural network is trained using backpropagation and gradient descent.

## Features
- **Abstract Neural Network Class**: The core of the project is an abstract neural network class that can be used for any classification task with configurable hidden layers.
- **Digit Recognition**: A script for recognizing digits (e.g., from EMNIST dataset).
- **Character Recognition**: A script for recognizing characters from the extended EMNIST dataset.
- **Pickle-based Serialization**: The weights and biases of the neural network are saved and loaded from a pickle file for persistence.

## Requirements

To install the necessary dependencies, you can use the following command:

pip install -r requirements.txt
You will need numpy for the neural network computations and pickle for saving/loading the model weights and biases.

requirements.txt
text
Copy
Edit
numpy
Installing EMNIST Dataset
You can download the EMNIST dataset from the official repository: EMNIST dataset.

Here are the steps to download and set up the dataset:

Go to the EMNIST dataset download page.
Download the dataset files (such as emnist-byclass.mat).
Extract the dataset files into a directory of your choice.
Load the dataset in your Python script using a library like scipy.io.loadmat or a similar method.
For example, you can load the data as follows:

python
Copy
Edit
from scipy.io import loadmat

Installing EMNIST Dataset
The EMNIST dataset can be downloaded from the official website. Follow the steps below to get the dataset:

Go to the EMNIST Dataset Page.
Choose the appropriate subset of the dataset (e.g., "EMNIST ByClass").
Download the .tar.gz file for the subset you want (e.g., emnist-byclass.tar.gz).
Extract the .tar.gz file and move it to a suitable directory.


