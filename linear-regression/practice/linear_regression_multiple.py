import numpy as np


x = np.array([[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]])
y = np.array([4, 5, 20, 14, 32, 22, 38, 43])

mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / std


ones = np.ones((x.shape[0], 1))
x = np.concatenate((ones, x), axis=1)


def gradient_descent(x, y, weights, learning_rate, num_iterations):
    for i in range(num_iterations):
        predictions = np.dot(x, weights)
        errors = predictions - y
        gradient = np.dot(x.T, errors) / len(y)
        weights -= learning_rate * gradient
    return weights


learning_rate = 0.01
num_iterations = 1000


weights = np.zeros(x.shape[1])
weights = gradient_descent(x, y, weights, learning_rate, num_iterations)


new_data = np.array([[10, 3], [20, 7]])
new_data = (new_data - mean) / std
new_data = np.concatenate((np.ones((new_data.shape[0], 1)), new_data), axis=1)


predictions = np.dot(new_data, weights)
print(predictions)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y, np.dot(x, weights))
print(mse)
