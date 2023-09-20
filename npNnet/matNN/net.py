from typing import List, Callable, Tuple, NewType
import numpy as np
import matplotlib.pyplot as plt

from layer import Layer
from datautils import shuffle_io, encode_categories
from optimizers import Optimizer

NetworkData = NewType("NetworkData", Tuple[np.ndarray, np.ndarray])
LOGS = False


class Network:
    def __init__(
        self,
        layers: List[Layer],
        loss: Callable,
        lr: float,
        optimizer: "Optimizer" = None,
        epochs=500,
    ) -> None:
        self.net = layers
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.opti = optimizer

    def fwd_pred(self, inputvec: np.ndarray) -> np.ndarray:
        out = inputvec
        for layer in self.net:
            out = layer.forward(out)

        return out

    def train(
        self,
        train_data: NetworkData,
        val_data: NetworkData = (None, None),
        shuffle=False,
    ):
        J = []
        J_validn = []
        x_train, y_train = train_data
        x_val, y_val = val_data
        lossfnc, loss_gradfnc = self.loss()

        for e in range(self.epochs):
            if shuffle:
                x_train, y_train = shuffle_io(x_train, y_train)
            epoch_err = 0.0

            # SGD
            for x, y in zip(x_train, y_train):
                ypred = self.fwd_pred(x)
                epoch_err += lossfnc(y, ypred)
                grad = loss_gradfnc(y, ypred)

                for layer in reversed(self.net):
                    grad = layer.backward(grad, self.lr, self.opti)

            epoch_err /= len(x_train)
            J.append(np.mean(epoch_err))

            if x_val is not None and y_val is not None:
                err = [lossfnc(y, self.fwd_pred(x)) for x, y in zip(x_val, y_val)]
                J_validn.append(np.sum(err) / len(x_val))

            if LOGS:
                print(e + 1, epoch_err)

        return J, J_validn

    def fwd_test(self, x_test):
        res = [self.fwd_pred(x) for x in x_test]
        return np.stack(res)


def plotlosses(train_loss: list, val_loss: list, other=None):
    plt.plot(train_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.ylim(0, 1)
    plt.title("Error Trend")

    if val_loss:
        plt.plot(val_loss, label="Validation Loss")

    plt.legend()
    return plt.show()
