# Two-layer, sigmoid feedforward network
# trained using the "Extreme Learning Machine" algorithm.
# Adapted from https://gist.github.com/larsmans/2493300

# TODO: make it possible to use alternative linear classifiers instead
# of pinv2, e.g. SGDRegressor
# TODO: implement partial_fit and incremental learning
# TODO: tr

import numpy as np
from scipy.linalg import pinv2

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.random_projection import sparse_random_matrix


def relu(X):
    """Rectified Linear Unit"""
    return np.clip(X, 0, None)


class ELMClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Extreme Learning Machine Classifier

    Basically a 1 hidden layer MLP with fixed random weights on the input
    to hidden layer.

    TODO: document parameters and fitted attributes.
    """

    activations = {
        'tanh': np.tanh,
        'relu': relu,
    }

    def __init__(self, n_hidden=1000, rank=None, activation='tanh',
                 random_state=None, density='auto'):
        self.n_hidden = n_hidden
        self.rank = rank
        if activation is not None and activation not in self.activations:
            raise ValueError(
                "Invalid activation=%r, expected one of: '%s' or None"
                % (activation, "', '".join(self.activations.keys())))
        self.activation = activation
        self.density = density
        self.random_state = random_state

    def fit(self, X, y):
        if self.activation is None:
            # Useful to quantify the impact of the non-linearity
            self._activate = lambda x: x
        else:
            self._activate = self.activations[self.activation]
        rng = check_random_state(self.random_state)

        # one-of-K coding for output values
        self.classes_ = unique_labels(y)
        Y = label_binarize(y, self.classes_)

        # set hidden layer parameters randomly
        n_features = X.shape[1]
        if self.rank is None:
            if self.density == 1:
                self.weights_ = rng.randn(n_features, self.n_hidden)
            else:
                self.weights_ = sparse_random_matrix(
                    self.n_hidden, n_features, density=self.density,
                    random_state=rng).T
        else:
            # Low rank weight matrix
            self.weights_u_ = rng.randn(n_features, self.rank)
            self.weights_v_ = rng.randn(self.rank, self.n_hidden)
        self.biases_ = rng.randn(self.n_hidden)

        # map the input data through the hidden layer
        H = self.transform(X)

        # fit the linear model on the hidden layer activation
        self.beta_ = np.dot(pinv2(H), Y)
        return self

    def transform(self, X):
        # compute hidden layer activation
        if hasattr(self, 'weights_u_') and hasattr(self, 'weights_v_'):
            projected = safe_sparse_dot(X, self.weights_u_, dense_output=True)
            projected = safe_sparse_dot(projected, self.weights_v_)
        else:
            projected = safe_sparse_dot(X, self.weights_, dense_output=True)
        return self._activate(projected + self.biases_)

    def decision_function(self, X):
        return np.dot(self.transform(X), self.beta_)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


if __name__ == "__main__":
    from sklearn.cross_validation import train_test_split
    from time import time

    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target

    # from sklearn.datasets import fetch_covtype
    # covtype = fetch_covtype()
    # X, y = covtype.data, covtype.target

    # from sklearn.datasets import fetch_20newsgroups_vectorized
    # twenty = fetch_20newsgroups_vectorized()
    # X, y = twenty.data, twenty.target
    # X = X[y < 4]
    # y = y[y < 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3, random_state=0)

    for n_hidden in [100, 200, 500, 1000, 2000, 5000, 10000]:
        print("Fitting ELM for n_hidden=%d..." % n_hidden)
        tic = time()
        model = ELMClassifier(n_hidden=n_hidden, rank=None, density='auto',
                              activation='relu')
        model.fit(X_train, y_train)
        toc = time()
        print("done in %0.3fs: train accuracy=%0.3f, test accuracy=%0.3f"
              % (toc - tic,
                 model.score(X_train, y_train),
                 model.score(X_test, y_test)))
