import numpy as np


# class for base layer
class Layer:
    def __init__(self, input: np.ndarray, output: np.ndarray):
        self.input = input
        self.output = output

    def forward(self, input: np.ndarray):
        # TODO return fwd output
        pass

    def backward(self, output_grad: np.ndarray, alpha: float):
        # TODO update params return d(err)/d(input) to prev layer
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):  # size = number of nodes
        self.weis = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.dot(self.weis, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)  # formula
        input_gradient = np.dot(self.weis.T, output_gradient)  # formula

        self.weis -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient  # return grad of this input as o/p grad to n-1 layer
