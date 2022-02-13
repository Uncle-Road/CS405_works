import numpy as np


class KNN:
    def __init__(self, K, metric_type):
        """
        Args:
            K (int): K-value
            metric_type (str): L1, L2, or L-inf
        """
        self.K = K
        self.metric_type = metric_type

    def distance_func(self, vec1, vec2):
        """
        Computes the distance between two d-dimension vectors.

        Args:
            vec1 ((d,) np.ndarray): d-dim vector
            vec2 ((d,)) np.ndarray): d-dim vector
        """
        diff = vec1 - vec2
        distance = 0
        if self.metric_type == "L1":
            # write your code here
            # todo
            distance = np.sum(np.abs(diff))

        elif self.metric_type == "L2":
            # write your code here
            distance = np.sqrt(np.sum(np.square(diff)))
        elif self.metric_type == "L-inf":
            # write your code here
            distance = np.max(np.abs(diff))

        return distance

    def fit(self, x_train, y_train):
        """
        Args:
            x_train ((n,d) np.ndarray): training data with n samples and d features
            y_train ((n,) np.ndarray): training labels
        """
        self.x_train = x_train
        self.y_train = y_train

    def compute_distances_neighbors(self, sample):
        """
        Computes the distance between every data point in the train set and the 
        given sample and then finds the k-nearest neighbors.

        Returns a numpy array of the labels of the k-nearest neighbors.

        Args:
            sample ((d,) np.ndarray): the given sample to be computed

        Returns:
            neighbors (list): K-nearest neighbors' labels
        """

        # write your code here
        # todo

        dis = []
        for i in range(self.x_train.shape[0]):
            dis.append(self.distance_func(sample, self.x_train[i, :]))
        labels = []
        index = sorted(range(len(dis)), key=dis.__getitem__)
        for j in range(self.K):
            labels.append(self.y_train[index[j]])
        neighbors = labels

        return neighbors

    @staticmethod
    def majority(neighbors):
        """
        Performs majority voting and returns the predicted value for the test sample.
        Since we're performing binary classification, the possible values are [0,1].

        Args:
            neighbors (list): K-nearest neighbors' labels

        Returns:
            predicted_value (int): the predicted label for the given sample
        """

        # write your code here
        # todo
        bins = np.bincount(neighbors)
        predicted_value = np.argmax(bins)
        return predicted_value

    def predict(self, X_test):
        """
        Computes the predicted values for the entire test set.

        Args:
            x_train ((n,d) np.ndarray): training data with n samples and d features
            y_train ((n,) np.ndarray): training labels
            X_test ((n,d) np.ndarray): test data

        Returns:
            pred_test ((n,) np.ndarray): output for every entry in the test set
        """
        pred_test = np.array(
            [self.majority(self.compute_distances_neighbors(sample)) for sample in X_test])

        return pred_test


def accuracy_score(pred, y):
    """
    Computes the accuracy of the predicted data.

    Args:
        pred ((n,) np.ndarray): predicted values for n samples
        y ((n,) np.ndarray): labels for n samples

    Returns:
        acc (float): accuracy
    """

    acc = np.mean(pred == y)
    # print(acc)
    return acc


def main():
    n, m, d = input().split()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for _ in range(int(n)):
        split_line = input().split()
        X = split_line[:int(d)]
        y = split_line[-1]
        x_train.append([float(x) for x in X])
        y_train.append(int(y))
    for _ in range(int(m)):
        split_line = input().split()
        X = split_line[:int(d)]
        y = split_line[-1]
        x_test.append([float(x) for x in X])
        y_test.append(int(y))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    metric_types = ["L1", "L2", "L-inf"]
    params = [(x, y) for x in range(1, 6, 1) for y in metric_types]
    value = -100
    list = []
    for param in params:
        model = KNN(param[0], param[1])
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(pred, y_test)
        # print(param)
        # print(acc)
        if acc > value:
            list.clear()
            list.append(param)
            value = acc
            continue
        elif acc == value:
            list.append(param)
            continue
        else:
            continue
    b = np.array(list)
    a = np.sort(b, kind='stable')
    for _ in a:
        print(_[0], end=" ")
        print(_[1])
    # print(1)

if __name__ == '__main__':
    main()
