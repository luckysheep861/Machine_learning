import numpy as np

class PCA:
    def __init__(self, n_components):
        assert n_components >= 1, "n_component must be valid"
        self.n_components = n_components
        self.components_ = None

    def __repr__(self):
        return "PCA(N_components = {})".format(self.n_components)

    def fit(self, X, eta=0.01, n_iters=1e4):
        assert self.n_components <= X.shape[1], \
            "n_componenet must be greater than the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            cur_iter = 0
            w = direction(initial_w)
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)  # 每次求一个单位方向
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta)
            self.components_[i,:] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X):
        assert self.components_ is not None, "must fit before transform"
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        assert self.components_ is not None, "must fit before transform"
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)