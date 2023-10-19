import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1e-2) -> None:
        self.learning_rate = learning_rate

    def get_update_term(self, param: np.ndarray, param_grad: np.ndarray):
        pass


class Momentum(Optimizer):
    def __init__(self, learning_rate=1e-2, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity = None

    def get_update_term(self, param: np.ndarray, param_grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(param)

        self.velocity = self.beta * self.velocity + (1 - self.beta) * param_grad
        return self.velocity * self.learning_rate


class Nesterov(Momentum):
    def get_update_term(self, param, param_grad):
        prev_velocity = super().get_update_term(param, param_grad)
        param_lookahead = param - self.beta * prev_velocity

        return (param - param_lookahead) + self.learning_rate * param_grad


class Adagrad(Optimizer):
    def __init__(self, learning_rate=1e-2, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.velocity = None

    def get_update_term(self, param: np.ndarray, param_grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(param)

        self.velocity += param_grad**2
        return self.learning_rate * param_grad / (self.velocity**0.5 + self.epsilon)


class RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-2, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.velocity = None

    def get_update_term(self, param: np.ndarray, param_grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(param)

        self.velocity = (
            self.beta * self.velocity + (1 - self.beta) * param_grad**2
        )  # use exp weighted average

        return self.learning_rate * param_grad / (self.velocity**0.5 + self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-2, b1=0.9, b2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.b1 = b1
        self.b2 = b2
        self.t = 0
        self.velocity = None
        self.momentum = None
        self.epsilon = epsilon

    def get_update_term(self, param: np.ndarray, param_grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.momentum is None:
            self.momentum = np.zeros_like(param)
        if self.velocity is None:
            self.velocity = np.zeros_like(param)

        # from momentum sgd
        self.momentum = self.b1 * self.momentum + (1 - self.b1) * param_grad
        # from rmsprop sgd
        self.velocity = self.b2 * self.velocity + (1 - self.b2) * (param_grad**2)

        momentum_corrected = self.momentum / (1 - self.b1**self.t)
        velocity_corrected = self.velocity / (1 - self.b2**self.t)

        return (
            self.learning_rate
            * momentum_corrected
            / (velocity_corrected**0.5 + self.epsilon)
        )
