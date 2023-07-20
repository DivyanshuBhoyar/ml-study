import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_model(X, y, learning_rate=0.01, num_iterations=1000):
    w = np.zeros(X.shape[1])
    b = 0.0

    for _ in range(num_iterations):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        cost = -1 / len(y) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        dw = 1 / len(y) * np.dot(X.T, (y_pred - y))
        db = 1 / len(y) * np.sum(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return np.round(y_pred)


def calculate_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def logistic_regression_train(
    X_train, y_train, learning_rate=0.01, num_iterations=1000
):
    X_train = X_train[:, np.newaxis]
    w, b = train_model(X_train, y_train, learning_rate, num_iterations)
    return w, b


def test_regression(X_test, w, b):
    X_test = X_test[:, np.newaxis]
    predictions = predict(X_test, w, b)
    return predictions


X_train = np.array([2, 3, 4, 5, 6, 7, 8, 9])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_test = np.array([10, 11, 12])
y_test = np.array([1, 1, 1])

w, b = logistic_regression_train(X_train, y_train)
train_predictions = test_regression(X_train, w, b)
test_predictions = test_regression(X_test, w, b)

train_accuracy = calculate_accuracy(train_predictions, y_train)
test_accuracy = calculate_accuracy(test_predictions, y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

study_hours = 7
prediction = predict(np.array([study_hours]), w, b)
if prediction == 1:
    print("The student is predicted to pass.")
else:
    print("The student is predicted to fail.")
