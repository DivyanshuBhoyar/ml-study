import numpy as np


class Layer:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def forward(self, ip: np.ndarray):
        pass

    def backward(self, ouput_grad: np.ndarray, alpha: float, optim):
        pass
