'''
As the name suggests, Pipeline
class allows sticking multiple processes into a single scikit-learn estimator. 
pipeline class has fit, predict and score method just like any other estimator 
(in this application SVC: kernel: 'rbf').

Parkinson's Data Set, provided by UCI's Machine Learning Repository. 
The dataset was created at the University of Oxford, in collaboration with 10 medical centers around the US, 
along with Intel who developed the device used to record the primary features of the dataset: speech signals.'''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
from time import sleep
import warnings
warnings.filterwarnings('ignore')

PCA_n_components_parameters = (np.arange(2, 15, 1))

SVM_c_parameters = (np.arange(0.05, 2.05, 0.05))

SVM_gamma_parameters = (np.arange(0.001, 0.101, 0.001))

X = pd.read_csv('parkinsons.data')

X.drop(['name'], axis=1, inplace=True)

y = X.status.copy()

X.drop(['status'], axis=1, inplace=True) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=7)

pipe_steps = [('Scaler', preprocessing.StandardScaler()), ('PCA', PCA()), ('SVM', SVC(kernel='rbf'))]

check_params = {'PCA__n_components': PCA_n_components_parameters,'SVM__C': SVM_c_parameters,'SVM__gamma': SVM_gamma_parameters}

pipeline = Pipeline(pipe_steps)

print()
print('*'*40)

progress_bar = tqdm(list(string.ascii_lowercase))
print('Starting Training Data')

for letter in progress_bar:
    create_grid = GridSearchCV(pipeline, param_grid=check_params)
    create_grid.fit(X_train, y_train)
    progress_bar.set_description(f'Processing {letter}...')
    sleep(0.09)
print('Score from Testing Data: ', create_grid.score(X_test, y_test))
print('Best Fit Parameters From Training Data', create_grid.best_params_)