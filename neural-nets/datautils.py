from random import shuffle
import numpy as np
from typing import List

from pandas import DataFrame
import numpy as np


# to shuffle the data in each epoch
def shuffle_io(x, y):
    combined_data = list(zip(x, y))
    shuffle(combined_data)
    x_train, y_train = zip(*combined_data)

    return x_train, y_train


def encode_categories(data: np.array, classes: List[str]) -> np.ndarray:
    encoded = np.zeros((len(data), len(classes)))
    class_lables = {name: idx for idx, name in enumerate(classes)}

    for i, label in enumerate(data):
        encoded[i, class_lables[label]] = 1

    return encoded


def my_train_test_split(df: DataFrame, y_columns, test_size, random_state=None):
    # shuffle
    df = df.sample(frac=1, random_state=random_state)

    x_columns = [col for col in df.columns if col not in y_columns]

    X = df[x_columns].to_numpy()
    Y = df[y_columns].to_numpy()

    split_index = int((1.0 - test_size) * len(df))

    # Finally, we split the data
    X_train = X[:split_index].reshape(split_index, len(x_columns), 1)
    Y_train = Y[:split_index].reshape(split_index, len(y_columns), 1)
    X_test = X[split_index:].reshape(len(df) - split_index, len(x_columns), 1)
    Y_test = Y[split_index:].reshape(len(df) - split_index, len(y_columns), 1)

    return X_train, Y_train, X_test, Y_test
