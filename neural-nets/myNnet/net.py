from typing import List
from .layer import Layer, Dropout
import numpy as np
from typing import Callable
from .datautils import shuffle_io, encode_categories


def pred(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


#  full batch
def train(
    network: List[Layer],
    X_train,
    Y_train,
    loss: Callable,
    loss_grad: Callable,
    logs=True,
    alpha=0.01,
    epochs=1000,
    shuffle=False,
):
    J = []  # error history
    if shuffle:
        X_train, Y_train = shuffle_io(X_train, Y_train)

    for e in range(epochs):
        err = 0

        for x, y in zip(X_train, Y_train):
            # forward
            # we are not reinitialising any params here, live updated params from prev will be used in fwd_pred
            output = pred(network, x)  # fwd

            err += loss(y, output)
            grad = loss_grad(y, output)  # dE/dYpred

            # bacward
            for layer in reversed(network):
                grad = layer.backward(grad, alpha)  # grad updates per layer

        err /= len(X_train)
        if logs:
            print("error for epoch ", e + 1, "= ", err)

        J.append(err)

    return J


def run_test(network, input):
    network = [layer for layer in network if not isinstance(layer, Dropout)]
    print(id(network[0]), id(network[2]))

    res = []
    for row in input:
        p = pred(network, row)
        res.append(p)

    return np.stack(res)


def evaluate_classifn(y_true, y_out):
    y_out = np.squeeze(y_out, axis=2)
    y_true = np.squeeze(y_true, axis=2)

    y_out = np.argmax(y_out, axis=1)
    y_true = np.argmax(y_true, axis=1)

    accuracy = np.mean(y_true == y_out)

    print("Network accuracy ☢️", accuracy)
