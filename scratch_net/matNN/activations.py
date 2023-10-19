import numpy as np
from activation import Activation
from layer import Layer


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)
        super().__init__(relu, relu_prime)


class SoftPlus(Activation):
    def __init__(self):
        softplus = lambda x: np.log(1 + np.exp(x))
        softplus_prime = lambda x: 1 / (1 + np.exp(-x))
        return super().__init__(softplus, softplus_prime)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        leaky_relu = lambda x: np.where(x >= 0, x, alpha * x)
        leaky_relu_prime = lambda x: np.where(x >= 0, 1, alpha)
        super().__init__(leaky_relu, leaky_relu_prime)


class ELU(Activation):
    def __init__(self, alpha=1.0):
        elu = lambda x: np.where(x > 0, x, alpha * (np.exp(x) - 1))
        elu_prime = lambda x: np.where(x > 0, 1, elu(x) + alpha)
        super().__init__(elu, elu_prime)


# only for the last layer
class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, err_gradient, learning_rate):
        n = np.size(self.output)
        # return np.dot((np.identity(n) - self.output.T) * self.output, err_gradient)

        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), err_gradient)
