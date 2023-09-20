import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate


class Momentum(Optimizer):
    def __init__(self, learning_rate=1e-2, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity = None

    def update(self, w, dw):
        if self.velocity is None:
            self.velocity = np.zeros_like(w)

        self.velocity = self.beta * self.velocity - self.learning_rate * dw
        next_w = w + self.velocity

        return next_w


class Nesterov(Momentum):
    def update(self, w, dw):
        v_prev = np.copy(self.velocity)
        super().update(w, dw)
        next_w = w - self.beta * v_prev + (1 + self.beta) * self.velocity

        return next_w


class Adagrad(Optimizer):
    def __init__(self, learning_rate=1e-2, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = None

    def update(self, w, dw):
        if self.cache is None:
            self.cache = np.zeros_like(w)

        self.cache += dw**2
        next_w = w - self.learning_rate * dw / (np.sqrt(self.cache) + self.epsilon)

        return next_w


class RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-2, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def update(self, w, dw):
        if self.cache is None:
            self.cache = np.zeros_like(w)

        self.cache = self.beta * self.cache + (1 - self.beta) * dw**2
        next_w = w - self.learning_rate * dw / (np.sqrt(self.cache) + self.epsilon)

        return next_w


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-2, b1=0.9, b2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, dw):
        if (self.m is None) or (self.v is None):
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1
        m_t_hat = None
        v_t_hat = None

        # Update biased first moment estimate
        self.m = (self.b1 * self.m) + ((1 - self.b1) * dw)

        # Compute bias-corrected first moment estimate
        m_t_hat = self.m / (1 - (self.b1**self.t))

        # Update biased second raw moment estimate
        v_t_hat = (self.b2 * self.v) + ((1 - self.b2) * (dw**2))

        # Compute bias-corrected second raw moment estimate
        v_t_hat = self.v / (1 - (self.b2**t))

        # Update parameters
        next_w = w - (self.learning_rate / (np.sqrt(v_t_hat) + self.epsilon)) * m_t_hat

        return next_w
