from typing import List
from layer import Layer
from losses import mse, mse_prime


def pred(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network: List[Layer], X_train, Y_train, logs=True, alpha=0.01, epochs=1000):
    for e in range(epochs):
        err = 0

        for x, y in zip(X_train, Y_train):
            # forward
            output = x
            for layer in network:
                output = layer.forward(
                    output
                )  # we are not reinitialising params here, live updated params from prev will be used as layers are defined only once
            # output = pred(network, x)

            err += mse(y, output)
            grad = mse_prime(y, output)

            # bacward
            for layer in reversed(network):
                grad = layer.backward(grad, alpha)

        if logs:
            err /= len(X_train)
            print("error for epoch ", e + 1, "= ", err)


# epochs = 1000
# alpha = 0.011

# for e in range(epochs):
#     err = 0

#     for x, y in zip(X, Y):
#         # forward
#         output = x
#         for layer in network:
#             output = layer.forward(output) # we are not reinitialising params here, live params from prev res will be used as layers are defined only once

#         err += mse(y, output)
#         grad = mse_prime(y, output)

#         #bacward
#         for layer in reversed(network):
#             grad = layer.backward(grad, alpha)

#     err /= len(X)
#     print("error for epoch ", e+1, "= ", err)
