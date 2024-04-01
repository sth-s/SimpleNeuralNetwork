## Introduction
This project is created to introduce the basics of neural networks. It demonstrates how to implement a simple neural network in Python and use it to solve the task of recognizing handwritten digits.

## Project Components
- `NeuralNetwork.py`: The main class implementing the neural network. This class represents a universal neural network, whose parameters (such as activation functions, learning rate, number of layers, etc.) can be configured. It supports saving and loading the neural network state.
- `FunctionStore.py`: A class that contains activation functions and their derivatives.
- `NumbersModelTrainer.py`: Demonstrates the training process of the neural network.
- `Painter.py`: An application for drawing and testing the neural network's digit recognition.
- `run.py`: The script to run the project.

## Neural Network Training
The training process of the neural network is demonstrated in the `NumbersModelTrainer.py` class. The network was trained using the dataset from [pjreddie's mnist-csv-png](https://github.com/pjreddie/mnist-csv-png).

## Testing the Neural Network
You can test the performance of the neural network yourself by downloading the dataset from the provided source or by extracting the archives in the repository and executing the script in the `run.py` file, after uncommenting the necessary part of the code. (don't forget to specify the correct path to the datasets in the script)

The repository also contains the config of the neural network, which showed more than 90% accuracy on both the training and test datasets. Achieving such a result took a very long time on my hardware, but I hope it will be sufficient to verify the functionality of the neural network.

You can visually check the performance of the neural network using `Painter.py`, launched through `run.py`. `Painter.py` allows you to draw digits in real-time and see how the neural network recognizes them.

## Installation and Running

Before running, ensure the following libraries are installed: Tkinter and PIL. They are usually installed with Python, but if you encounter any issues, try the following:

```bash
pip install numpy pillow
```

or

```bash
sudo apt-get install python3-tk python3-pil python3-pil.imagetk
```

To run the project, execute run.py.