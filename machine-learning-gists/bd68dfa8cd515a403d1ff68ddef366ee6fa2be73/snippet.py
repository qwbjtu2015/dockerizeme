#!/usr/bin/env python

import urllib2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    X = []
    Y = []
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    for line in urllib2.urlopen(data_url).readlines():
        line = map(float, line.split())
        X.append(line[0:13])
        Y.append(line[13])
    return X, Y


def basic_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def deeper_model():
    # create model
    model = Sequential()
    model.add(Dense(13, kernel_initializer='normal', activation='relu', input_dim=13))
    model.add(Dense(6,  kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,  kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train(X, Y, fn, standardize=True, seed=7):
    np.random.seed(seed)
    estimators = []
    if standardize:
        estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=fn, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print('Result: %.2f (%.2f) MSE' % (results.mean(), results.std()))



if __name__ == '__main__':
    X, Y = load_data()
    train(X, Y, fn=basic_model,  standardize=False, seed=7)
    train(X, Y, fn=basic_model,  standardize=True,  seed=7)
    train(X, Y, fn=deeper_model, standardize=True,  seed=7)
    train(X, Y, fn=wider_model,  standardize=True,  seed=7)