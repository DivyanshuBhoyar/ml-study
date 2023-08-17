import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activtn, activtn_prime):
        self.activtn = activtn
        self.activtn_prime = activtn_prime

    # gets linear z as input, spits out activated a during fwd prop
    def forward(self, input: np.ndarray):
        self.input = input
        return self.activtn(self.input)

    # input is output grad
    def backward(self, output_grad: np.ndarray, alpha: float):
        return np.multiply(
            output_grad, self.activtn_prime(self.input)
        )  # based on the formula


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


class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def softmax_prime(self, x):
        p = self.softmax(x)
        return p * (1 - p)


class SoftPlus(Activation):
    def __init__(self):
        softplus = lambda x: np.log(1 + np.exp(x))
        softplus_prime = lambda x: 1 / (1 + np.exp(-x))
        return super().__init__(softplus, softplus_prime)
