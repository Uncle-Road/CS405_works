import numpy as np
import itertools
import functools


class PolynomialFeatures:
    """
    polynomial features
    transforms input array with polynomial features
    Example
    =======
    x =
    [[a, b],
    [c, d]]
    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features
        Parameters
        ----------
        degree : int
        degree of polynomial
        """
        self.degree = degree

    def fit_transform(self, X):
        """
        transforms input array with polynomial features
        Parameters
        ----------
        X : (sample_size, n) ndarray
        input array
        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
        polynomial features
        """
        features = [np.ones(len(X))]
        X = X[np.newaxis, :]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(X, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()

class BayesianRegression:
    """
    Bayesian regression model

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha: float = 1., beta: float = 1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim: int) -> tuple:
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        bayesian update of parameters given training dataset
        Parameters
        5
        ----------
        X : (N, n_features) np.ndarray
        training data independent variable
        t : (N,) np.ndarray
        training data dependent variable
        """
        mean_prev, precision_prev = self._get_prior(np.size(X, 1))
        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        """
        return mean and standard deviation of predictive distribution

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            independent variable

        Returns
        -------
        y : (N,) np.ndarray
            mean of the predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of the predictive distribution
        """
        # write your code here
        #todo
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = X @ w_sample.T
            return y_sample
        y = X @ self.w_mean

        y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
        y_std = np.sqrt(y_var)
        return y, y_std


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


def main():
    n, m = input().split()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for _ in range(int(n)):
        x, y = input().split()
        x_train.append(float(x))
        y_train.append(float(y))
    for _ in range(int(m)):
        x, y = input().split()
        x_test.append(float(x))
        y_test.append(float(y))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # write your code here
    #todo
    model = PolynomialFeatures(degree=10)
    x_transform_train = model.fit_transform(x_train)
    x_transform_test = model.fit_transform(x_test)
    bayes=BayesianRegression()
    bayes.fit(x_transform_train,y_train)
    y_predict_test,y_std_test = bayes.predict(x_transform_test)
    y_predict_train = bayes.predict(x_transform_train)
    print("{0:.6f}".format(rmse(y_predict_test,y_test)))
    for lines in y_std_test:
        print(round(lines,6))
    # print(rmse(y_predict_train,y_train))



if __name__ == '__main__':
    main()
