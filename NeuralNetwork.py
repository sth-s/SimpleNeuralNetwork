from __future__ import annotations
import numpy as np
import json
from FunctionStore import FunctionStore

class NeuralNetwork:
    def __init__(self, learning_rate, activation="sigmoid", derivative="sigmoid_derivative", *sizes, predefined_weights=None, predefined_biases=None):
        self.learning_rate = learning_rate

        # Define activation and derivative
        self.func_store = FunctionStore()
        try:
            if isinstance(activation, str):
                activation = self.func_store.call(activation)
            self.pre_activation = activation
            self.activation = np.vectorize(self.pre_activation)

            if isinstance(derivative, str):
                derivative = self.func_store.call_derivative(derivative)
            self.pre_derivative = derivative
            self.derivative = np.vectorize(self.pre_derivative)
        except ValueError as e:
            print(e)
            raise 

        # Define weights and biases
        self.sizes = sizes
        if predefined_weights is not None:
            self.weights = predefined_weights
        else:
            self.weights = [np.random.rand(y, x) * 2 - 1 for x, y in zip(sizes[:-1], sizes[1:])]
        if predefined_biases is not None:
            self.biases = predefined_biases
        else:
            self.biases = [np.random.rand(y) * 2 - 1 for y in sizes[1:]]

    def feed_forward(self, inputs):
        # Forward feed
        inputs = np.array(inputs)
        for weights, biases in zip(self.weights, self.biases):
            inputs = self.activation(np.dot(weights, inputs) + biases)
        return inputs

    def backpropagation(self, inputs, targets):
        layer_activations = [np.array(inputs)]  # activated inputs
        pre_activation_values = []  # weighted inputs

        # Forward feed
        for weights, biases in zip(self.weights, self.biases):
            weighted_inputs = np.dot(weights, layer_activations[-1]) + biases
            pre_activation_values.append(weighted_inputs)
            layer_activations.append(self.activation(weighted_inputs))
        
        # Backward pass
        error = (layer_activations[-1] - targets) * self.derivative(pre_activation_values[-1])
        gradients_b = [error]
        gradients_w = [np.outer(error, layer_activations[-2])]
        
        for l in range(2, len(self.sizes)):
            error = np.dot(self.weights[-l+1].T, error) * self.derivative(pre_activation_values[-l])
            gradients_b.append(error)
            gradients_w.append(np.outer(error, layer_activations[-l-1]))
        
        # Update weights and biases
        self.weights = [w-(self.learning_rate*gw) for w, gw in zip(self.weights, reversed(gradients_w))]
        self.biases = [b-(self.learning_rate*gb) for b, gb in zip(self.biases, reversed(gradients_b))]

    # Save NeutalNetwork config
    @staticmethod
    def save_nn_config(nn: NeuralNetwork, file_path):
        try:
            config = {
                "learning_rate": nn.learning_rate,
                "sizes": nn.sizes,
                "activation": nn.func_store.get_function_name(nn.pre_activation),
                "derivative": nn.func_store.get_derivative_name(nn.pre_derivative),
                "weights": [w.tolist() for w in nn.weights],
                "biases": [b.tolist() for b in nn.biases]
            }
        except ValueError as e:
            print(f"{e}\nFunction not found, save as unknown!")
            config = {
                "learning_rate": nn.learning_rate,
                "sizes": nn.sizes,
                "activation": "Unknown",
                "derivative": "Unknown",
                "weights": [w.tolist() for w in nn.weights],
                "biases": [b.tolist() for b in nn.biases]
            }
            
        with open(file_path, 'w') as f:
            json.dump(config, f)

    # Load NeuralNetwork from config
    @staticmethod
    def load_nn_config(file_path) -> NeuralNetwork:
        with open(file_path, 'r') as f:
            config = json.load(f)

        weights = [np.array(w) for w in config["weights"]]
        biases = [np.array(b) for b in config["biases"]]
        
        try:
            nn = NeuralNetwork(config["learning_rate"], config["activation"], config["derivative"], *config["sizes"],
                                predefined_weights=weights, predefined_biases=biases)
            return nn
        except ValueError as e:
            raise
