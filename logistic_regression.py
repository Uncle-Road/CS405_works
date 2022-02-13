import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        """
        output = (X - mean) / std

        """
        mean_X = X - self.mean_
        # std may contain zeros
        output = np.divide(
            mean_X, self.std_,
            out=np.zeros_like(mean_X), where=self.std_ != 0
        )
        return output


class LogisticRegressionAdaGrad:
    def __init__(self, num_epochs, learning_rate):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.w = None

    @staticmethod
    def sigmoid(z):
        """
        Calculates the sigmoid activation.

        """
        res = 1 / (1 + np.exp(-z))
        return res
        # write your code here

    def batch_loss(self, X, Y):
        """
        Calculates the averge loss of logistic regression over all training samples.

        """

        sig = self.sigmoid(np.dot(X, self.w))
        return (-Y @ np.log(sig) - (1 - Y) @ np.log(1 - sig)) / Y.size
        # write your code here

    def batch_gradient(self, X, Y):
        """
        Calculates the averge gradients of the weights over all training samples.

        """
        sig = self.sigmoid(np.dot(X, self.w))
        return np.dot((sig - Y), X)
        # write your code here

    def fit(self, X, Y):
        """
        Optimizes the weights using batch gradient descent with momentum (AdaGrad).
        Formula:
            m_{t+1} = m_t + square(grad_t)
            w_{t+1} = w_t - lr / (sqrt(m_{t+1}) + e) * grad_t

        where w_0 = [1, 1, ...], m_0 = [0, 0, ...], and e = 1e-8.

        """
        self.w = np.ones(X.shape[1])
        mom = np.zeros(X.shape[1])
        for i in range(self.num_epochs):
            grad = self.batch_gradient(X,Y)
            mom += np.square(grad)
            self.w -= (self.learning_rate / (np.sqrt(mom) + 1e-8)) * grad

        # write your code here

    def predict_proba(self, X):
        """
        Predicts the probabilities of the output labels.

        """
        y_prob = self.sigmoid(np.dot(X, self.w))
        return y_prob

    def predict(self, X):
        """
        Predicts the binary output labels using threshold 0.5.

        """
        y_pred = self.predict_proba(X)
        y_pred = y_pred > 0.5
        return y_pred




def accuracy_score(pred, y):
    return np.mean(pred == y)


def main():
    n, m, d = input().split()
    d = int(d)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for _ in range(int(n)):
        split_line = input().split()
        X = split_line[:d]
        y = split_line[-1]
        X_train.append([int(x) for x in X])
        y_train.append(int(y))
    for _ in range(int(m)):
        split_line = input().split()
        X = split_line[:d]
        y = split_line[-1]
        X_test.append([int(x) for x in X])
        y_test.append(int(y))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegressionAdaGrad(learning_rate=0.01, num_epochs=5000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(round(clf.batch_loss(X_train, y_train), 6))
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
