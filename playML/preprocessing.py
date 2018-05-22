import numpy as np

class StandarScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
        return self

    def tranform(self, X):
        assert self.mean_ is not None and self.scale_ is not None,\
               "must fit before transform!"
        assert X.shape[1] == len(self.mean_),\
               "the feature number of X must be qual to mean_ and std_"
        
        resX = np.empty(shape = X.shape, dtype = float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
        return resX
