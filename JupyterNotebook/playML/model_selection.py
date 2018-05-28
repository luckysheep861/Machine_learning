import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):

    assert X.shape[0] == y.shape[0]
    
    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_indexes = shuffled_indexes[test_size:]
    test_indexes = shuffled_indexes[:test_size]

    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]
