import numpy as np
from optimizers import Optimizer
from layer import Layer


class Dense(Layer):
    def __init__(self, in_size, out_size):
        self.weis = np.random.randn(out_size, in_size)
        self.bias = np.random.randn(out_size, 1)

    def forward(self, ip: np.ndarray):
        self.input = ip
        return np.dot(self.weis, self.input) + self.bias

    def backward(self, ouput_grad, alpha, optim: "Optimizer"):
        del_w = np.dot(ouput_grad, self.input.T)
        del_x = np.dot(self.weis.T, ouput_grad)
        # del_b = ouput_grad

        self.weis -= (
            optim.get_update_term(self.weis, del_w)
            if optim is not None
            else alpha * del_w
        )
        self.bias -= (
            optim.get_update_term(self.bias, ouput_grad)
            if optim is not None
            else alpha * ouput_grad
        )

        return del_x
