import numpy as np

# Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

class FunctionStore:
    def __init__(self):
        # Map Name to Func
        self.functions = {
            "sigmoid": sigmoid,
            "relu": relu
        }
        self.derivatives = {
            "sigmoid_derivative": sigmoid_derivative,
            "relu_derivative": relu_derivative
        }

    # Return fuction to each name
    def call(self, name):
        if name in self.functions:
            return self.functions[name]
        raise ValueError(f"Function '{name}' not found")
    
    # Return derivative to each name
    def call_derivative(self, name):
        if name in self.derivatives:
            return self.derivatives[name]
        raise ValueError(f"Derivative for '{name}' not found")
    
    def get_function_name(self, func):
        for name, registered_func in self.functions.items():
            if func == registered_func:
                return name
        raise ValueError("Function not found in store")

    def get_derivative_name(self, func):
        for name, registered_der in self.derivatives.items():
            if func == registered_der:
                return name
        raise ValueError("Function not found in store")
