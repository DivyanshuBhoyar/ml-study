from typing import Callable

from layer import Layer
import numpy as np


class Activation(Layer):
    def __init__(self, act_fn: Callable, act_prime: Callable):
        self.act_fn = act_fn
        self.act_prime = act_prime

    # takes linear z as input, gives 'a'
    def forward(self, ip: np.ndarray):
        self.input = ip
        return self.act_fn(self.input)

    def backward(self, output_grad: np.ndarray, alpha):
        return np.multiply(output_grad, self.act_prime(self.input))
