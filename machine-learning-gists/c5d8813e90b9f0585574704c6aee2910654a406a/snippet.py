#Handwritten digits datasets, such as The MNIST Database of handwritten digits, and Handwritten Digit Recognition to see how good you can get your classifier to perform on them.
#The MNIST problem is a dataset developed by Yann LeCun, Corinna Cortes and Christopher Burges for evaluating machine learning models on the handwritten digit classification problem.
#The dataset was constructed from a number of scanned document dataset available from the National Institute of Standards and Technology (NIST). This is where the name for the dataset comes from, as the Modified NIST or MNIST dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from array import array
import struct
from sklearn.model_selection import train_test_split
from random import*

def load(path_img, path_lbl):

    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {0}'.format(magic))
        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {0}'.format(magic))
        image_data = array("B", file.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    # You can set divisor to any int, e.g. 1, 2, 3. If you set it to 1,
    # there will be no resampling of the image. If you set it to two or higher,
    # the image will be resamples by that factor of pixels. This, in turn,
    # speeds up training but may reduce overall accuracy.
    divisor = 1
    for i in range(size):
        images[i] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)[::divisor,
                    ::divisor].reshape(-1)
    return pd.DataFrame(images), pd.Series(labels)

def peekData(X_train):
    # The 'targets' or labels are stored in y. The 'samples' or data is stored in X
    print("Peeking data")
    fig = plt.figure()
    cnt = 0
    for col in range(5):
        for row in range(10):
            plt.subplot(5, 10, cnt + 1)
            plt.imshow(X_train.iloc[cnt, :].values.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.axis('off')
            cnt += 1
    fig.set_tight_layout(True)
    return plt.show()


def drawPredictions(model,X,y):
    fig = plt.figure()
    # Making some guesses
    y_guess = model.predict(X)
    num_rows = 10
    num_cols = 5
    index = 0
    for col in range(num_cols):
        for row in range(num_rows):
            plt.subplot(num_cols, num_rows, index + 1)
            # 28x28 is the size of the image, 784 pixels
            plt.imshow(X.iloc[index, :].values.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
            # Green = Prediction right
            # Red = Fail!
            if y[index] == y_guess[index]:
                colors = 'green'
            else:
                colors = 'red'
            plt.title('Prediction: %i' % y_guess[index], fontsize=6, color=colors)
            plt.axis('off')
            index += 1
    fig.set_tight_layout(True)
    plt.show()
    return

def comparative (number):

    # : Print out the TRUE value of the digit in the test set
    # By TRUE value, we mean, the actual provided label for that sample

    true_value = y[number]
    print(f'{number}th Label: {true_value}')

    # : Predicting the value of the digits in the data set.
    # Why the model's prediction was incorrect?

    guess_Value = svc.predict(X[number:(number+1)])
    print(f'{number}th Prediction: {guess_Value}')
    print('#'*35)

    # : Using IMSHOW to display the imagez, so we can
    # visually check if it was a hard or easy image
    #
    plt.imshow(X.iloc[number, :].values.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'{number}th Number = Prediction: {guess_Value} x Label: {true_value}')
    return plt.show()

def prediction(y_guess, k,y):
    PredictionNotCorrect = []
    for index1 in range(0, k):
        if y[index1] != y_guess[index1]:
            PredictionNotCorrect.append(index1)
    PredictionNotCorrect1 = sample(PredictionNotCorrect, 2)

    for i in range(len(PredictionNotCorrect1)):
        TrueandFalse = comparative(PredictionNotCorrect1[i])

    return

X, y = load('train-MNIST.data', 'train-MNIST.labels')


# : Spliting data into test / train sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=7)

# Get to know the data.
peekData(X_train)
print('#'*35)

# : Creating a SVC classifier - Linear Kernel
print("Training SVC Classifier...")
svc = svm.SVC(kernel='linear', C=1, gamma=0.001)
svc.fit(X_train, y_train)
# : Calculating score of the SVC against TESTING data
print("Scoring SVC linear Classifier...")
score = svc.score(X_test, y_test)
print("Score: ", score)

# Confirmation of accuracy by 2 image
drawPredictions(svc, X, y)
size = y.size
y_guess = svc.predict(X)
prediction1 = prediction(y_guess,size, y)
print('#'*35)

# : Changing SVC classifier - Poly Kernel
svc = svm.SVC(kernel='poly', C=1, gamma=0.001)
svc.fit(X_train, y_train)
# : Calculating score of the SVC against TESTING data
print("Scoring SVC poly Classifier...")
score = svc.score(X_test, y_test)
print("Score: ", score)
# Confirmation of accuracy by 2 image
drawPredictions(svc, X, y)
y_guess = svc.predict(X)
prediction2 = prediction(y_guess,size,y)
print('#'*35)

# Changing  SVC classifier - RBF Kernel
svc = svm.SVC(kernel='rbf', C=1, gamma=0.001)
svc.fit(X_train, y_train)
# : Calculating score of the SVC against TESTING data
print("Scoring SVC rbf Classifier...")
score = svc.score(X_test, y_test)
print("Score: ", score)
# Visual Confirmation of accuracy
drawPredictions(svc, X, y)
y_guess = svc.predict(X)
prediction3 = prediction(y_guess,size, y)
print(prediction3)
print('#'*35)