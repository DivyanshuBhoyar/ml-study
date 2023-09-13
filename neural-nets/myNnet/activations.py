import numpy as np
from myNnet.layer import Layer


class Activation(Layer):
    def __init__(self, activtn, activtn_prime):
        self.activtn = activtn
        self.activtn_prime = activtn_prime

    # gets linear z as input, spits out activated `a` during fwd prop
    def forward(self, input: np.ndarray):
        self.input = input
        return self.activtn(self.input)

    # input is output grad; returns input gradient
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


class SoftPlus(Activation):
    def __init__(self):
        softplus = lambda x: np.log(1 + np.exp(x))
        softplus_prime = lambda x: 1 / (1 + np.exp(-x))
        return super().__init__(softplus, softplus_prime)


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
