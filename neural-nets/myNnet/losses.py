import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true: np.ndarray, y_pred: np.ndarray):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray):
    return -np.mean(y_true * np.log(y_pred))


def categorical_crossentropy_prime(y_true: np.ndarray, y_pred: np.ndarray):
    return y_pred - y_true


# OR
# return - y_true / y_pred
