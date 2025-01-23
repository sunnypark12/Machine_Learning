import numpy as np


def create_nl_feature(X):
    """
    TODO - Create additional features and add it to the dataset

    returns:
        X_new - (N, d + num_new_features) array with
                additional features added to X such that it
                can classify the points in the dataset.
    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    X_new = np.column_stack((X, x1 ** 2, x2 ** 2, x1 * x2))
    
    return X_new
