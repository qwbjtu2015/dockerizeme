# Daniel J. Rodriguez
# https://github.com/danieljoserodriguez
import numpy as np


# Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a 
# probability value between 0 and 1
def cross_entropy(y_hat, y):
    return -np.log(y_hat) if y == 1 else -np.log(1 - y_hat)

  
# Used for classification
def hinge(y_hat, y):
    return np.max(0, 1 - y_hat * y)

  
# Typically used for regression. It’s less sensitive to outliers than the MSE as it treats error as square only 
# inside an interval
def huber(y_hat, y, delta: int=1):
    return np.where(np.abs(y - y_hat) < delta, .5 * (y - y_hat) ** 2, delta * (np.abs(y - y_hat) - 0.5 * delta))

  
# Mean Squared Error, or L2 loss.
def mean_square_error(true_values, predicted_values):
    return ((true_values * predicted_values) ** 2.0).mean()

  
# Mean Absolute Error, or L1 loss.
def mean_absolute_error(y_hat, y):
    return np.sum(np.absolute(y_hat - y))
