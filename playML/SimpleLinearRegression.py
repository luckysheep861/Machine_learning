import numpy as np

class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim ==1, "x_train.ndim must equal to 1"
        assert len(x_train) == len(y_train), \
               "the size of x_train must equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x,y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a = num / d
        self.b = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        #x_predict,是一个一位数组
        assert x_predict.ndim = 1, \
               "this alg can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
               "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegresson1()"