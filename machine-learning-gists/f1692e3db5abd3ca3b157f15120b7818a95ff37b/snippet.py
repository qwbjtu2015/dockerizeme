'''
As the name suggests, Pipeline
class allows sticking multiple processes into a single scikit-learn estimator. 
pipeline class has fit, predict and score method just like any other estimator 
(in this application SVC: kernel: 'rbf').
Parkinson's Data Set, provided by UCI's Machine Learning Repository. 
The dataset was created at the University of Oxford, in collaboration with 10 medical centers around the US, 
along with Intel who developed the device used to record the primary features of the dataset: speech signals.'''

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import manifold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import string
from tqdm import tqdm
from time import sleep
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def WhatIsTheBestCV (BestCV):
    #progress_bar = tqdm(list(string.ascii_lowercase))
    print('Starting Training Data')
    Fold = 1
    cv1 = BestCV+2
    higherscore = 0
    TheBestCV = 0
    for i in range (2,cv1):
        progress_bar = tqdm(list(string.ascii_lowercase))
        for letter in progress_bar:
            scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
            #create_grid = GridSearchCV(pipeline, param_grid=check_params,cv=i, scoring='accuracy')
            create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=i, scoring=scoring, refit='AUC', return_train_score = True)
            create_grid.fit(X_train, y_train)
            #progress_bar.set_description(f'Processing {letter}...')
            sleep(0.09)
        print(f'CV = {Fold}')
        print('Score from Testing Data: ', create_grid.score(X_test,y_test))
        print('Best Fit Parameters From Training Data', create_grid.best_params_)
        print()
        print('*' * 40)
        k = create_grid.score(X_test, y_test)
        if higherscore <= k:
            higherscore = k
            TheBestCV = Fold
        Fold += 1
        progress_bar.set_description(f'Processing {letter}...')

    return higherscore, TheBestCV


def Parameters (a,b,c):
    Parameters1.clear()
    while True:
        Parameters1.append(a)
        a+=c
        if a >= b:
            break
    return Parameters1

check_params={}
Parameters1=[]

ISO_n_neighbors_parameters = Parameters(2, 6, 1)
check_params['ISO__n_neighbors'] = ISO_n_neighbors_parameters.copy()

ISO_n_components_parameters = Parameters(2, 8, 1)
check_params['ISO__n_components'] = ISO_n_components_parameters.copy()

SVM_c_parameters = Parameters(0.10, 0.105, 0.05)
check_params['SVM__C'] = SVM_c_parameters.copy()

SVM_gamma_parameters = Parameters(0.010, 0.105, 0.005)
check_params['SVM__gamma'] = SVM_gamma_parameters.copy()

Fold =((len(check_params['SVM__gamma']))*(len(check_params['SVM__C']))*(len(check_params['ISO__n_components']))*(len(check_params['ISO__n_neighbors'])))

X = pd.read_csv('parkinsons.data')

X.drop(['name'], axis=1, inplace=True)

y = X.status.copy()

X.drop(['status'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=7)

pipe_steps = [('Scaler', preprocessing.StandardScaler()), ('ISO', manifold.Isomap(max_iter=4)),('SVM', SVC(kernel='rbf'))]

pipeline = Pipeline(pipe_steps)

print()
print('*'*40)

BestScore,BestCV = WhatIsTheBestCV(Fold)
print(f'{BestCV} Fold (CV)')
print('Best Score: ', BestScore)