# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:32:47 2017

Supervised machine learning with scikit learn

L'objectif de ce tutoriel est de vous introduire à la manipulation de scikit-learn. 
Pour cela, on charge en mémoire de la donnée avant de voir comment mettre sur pied des modèles de machine learning supervisés. 
On verra ensuite comment ensembler les modèles, puis comment utiliser les procédures de cross-validation, comment utiliser la fonctionnalité de grid search (ainsi qu'un module supplémentaire s'intégrant très bien à scikit-learn, qui permet de faire du grid search évolutionnaire), comment utiliser la fonctionnalité d'élimination récursive des features.
Notez que le but de ce tutoriel est de comprendre comment un objet "modèle" peut être passé pour être transformé en objet "modèle cross validé" puis à nouveau passé pour être transformé en objet "modèle cross validé optimisé via grid search", puis à nouveau pour être transformé en "modèle ensemblé résultant de plusieurs modèles qui ont été cross validé et optimisé par une procédure de grid search". Bref, comprendre le flow, mettre le pied à l'étrier pour être à l'aise lors de vos premières utilisations de scikit-learn. 
Mais l'objectif n'est pas de tuner au mieux nos modèles, ni d'être exhaustif sur tous les modèles existants dans scikit-learn, ni d'expliquer ces modèles.

@author: FrançoisMalaussena
"""

# load dataset
from sklearn import datasets
cancer = datasets.load_breast_cancer()
cancer.data

# dataset into pd.dataframe
import pandas as pd
donnee = pd.concat([pd.DataFrame(data = cancer.data, columns = cancer.feature_names), 
                   pd.DataFrame(data = cancer.target, columns = ["target"])
                      ], axis = 1)

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(donnee.loc[:, donnee.columns != "target"], donnee.target, test_size = 0.25)

# this will be useful to see what models produced without rewriting the same code each time
models = []
def see_models(): 
    for model in models:
        print()
        print(model)
        print("Train set score :", model.score(X_train, y_train))
        print("Test set score :", model.score(X_test, y_test))

# let's try multiple models

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
models.append(classifier.fit(X_train, y_train))

from sklearn.linear_model import PassiveAggressiveClassifier #passive agressive classifier
classifier = PassiveAggressiveClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.svm import SVC #support vector machines
classifier = SVC()
models.append(classifier.fit(X_train, y_train))

from sklearn.svm import NuSVC
classifier = NuSVC()
models.append(classifier.fit(X_train, y_train))

from sklearn.svm import LinearSVC
classifier = LinearSVC()
models.append(classifier.fit(X_train, y_train))

from sklearn.linear_model import SGDClassifier #stochastic gradient descent
classifier = SGDClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.naive_bayes import GaussianNB # gaussian naive bayes
classifier = GaussianNB()
models.append(classifier.fit(X_train, y_train))

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
models.append(classifier.fit(X_train, y_train))

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
models.append(classifier.fit(X_train, y_train))

from sklearn.neighbors import KNeighborsClassifier # KNN
classifier = KNeighborsClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.neighbors.nearest_centroid import NearestCentroid
classifier = NearestCentroid()
models.append(classifier.fit(X_train, y_train))

from sklearn.gaussian_process import GaussianProcessClassifier # gaussian process
classifier = GaussianProcessClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.tree import DecisionTreeClassifier # decision trees. For interesting tree vizualisation, see graphviz module
classifier = DecisionTreeClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.ensemble import BaggingClassifier # bagging meta classifier
classifier = BaggingClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.ensemble import RandomForestClassifier # everyone's favorite homeboy random forest
classifier = RandomForestClassifier(max_features = 0.5, n_estimators = 1000, n_jobs = -1, verbose = 1)
models.append(classifier.fit(X_train, y_train))

from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.ensemble import AdaBoostClassifier # adaboost
classifier = AdaBoostClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.ensemble import GradientBoostingClassifier # gradient boosting
classifier = GradientBoostingClassifier()
models.append(classifier.fit(X_train, y_train))

from sklearn.neural_network import MLPClassifier #multi layer perceptron
classifier = MLPClassifier(hidden_layer_sizes = (100,100,100,100))
models.append(classifier.fit(X_train, y_train))

# let's see the results
see_models()

# let's try voting ensembling
from sklearn.ensemble import VotingClassifier
classifier1 = MLPClassifier(hidden_layer_sizes = (100,100,100,100))
classifier2 = AdaBoostClassifier()
classifier3 = ExtraTreesClassifier()
classifier = VotingClassifier(estimators=[
        ('mlp', classifier1), ('adaboost', classifier2), ('extratrees', classifier3)
        ], voting='soft', weights=[2, 1, 2])
models.append(classifier.fit(X_train, y_train))

# results
see_models()
models = []

# in the previous models, you were to optimize for the highest score on the test set
# but this was without using cross validation procedure. This is not a good idea. 
# read 3.1.0 : http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

#### cross validation
# let's use only one model
from sklearn.ensemble import AdaBoostClassifier # adaboost
classifier = AdaBoostClassifier()

# simple cross validation
from sklearn.model_selection import cross_validate
crossvalidated = cross_validate(classifier,
                                 donnee.loc[:, donnee.columns != "target"],
                                 donnee.target,                                
                                 cv = 5)
crossvalidated
crossvalidated.get("test_score").mean()

# example with KFold cross validation
from sklearn.model_selection import KFold
crossval_method = KFold(n_splits=3)
crossvalidated = cross_validate(classifier,
                                 donnee.loc[:, donnee.columns != "target"],
                                 donnee.target,                                
                                 cv = crossval_method)
crossvalidated.get("test_score").mean()

# run one of these and then run crossvalidated at the end
from sklearn.model_selection import RepeatedKFold
crossval_method = RepeatedKFold(n_splits=2, n_repeats=2)
from sklearn.model_selection import LeaveOneOut
crossval_method = LeaveOneOut()
from sklearn.model_selection import LeavePOut
crossval_method = LeavePOut(p = 1)
from sklearn.model_selection import ShuffleSplit
crossval_method = ShuffleSplit(n_splits=3, test_size=0.3)
from sklearn.model_selection import StratifiedKFold
crossval_method = StratifiedKFold(n_splits=3)

crossvalidated = cross_validate(classifier,
                                 donnee.loc[:, donnee.columns != "target"],
                                 donnee.target,                                
                                 cv = crossval_method)
crossvalidated.get("test_score").mean()

# see also
# from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit, TimeSeriesSplit

# from now on, with cross validation, train and test sets will be built from X_train and y_train. X_test and y_test are now really validation sets.
# therefore, you should optimize the features and the hyperparameters of your model for the best crossvalidated mean, when fitted on X_train, y_train
# 
# let's redefine see_models() to indicate this in the print statement
def see_models(): 
    for model in models:
        print()
        print(model)
        print("Train set score :", model.score(X_train, y_train))
        print("Validation set score :", model.score(X_test, y_test)) # the string changed

# however, testing repeteadly for different parameters of your model is a long process if you rewrite everything each time : let's introduce grid search

#### grid search
paramgrid = { 
        "activation" : ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs","sgd","adam"],
        "learning_rate":["constant","invscaling","adaptive"],
        "hidden_layer_sizes": [(100,80,50),(100,100,100),(50,50,50)]
        } 

# 4*3*3*3 combinations of parameters = 108 models will be tried

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(estimator = MLPClassifier(), 
                   param_grid = paramgrid, 
                   cv = StratifiedKFold(n_splits=4),
                   verbose = 1
                   )
%time models.append(cv.fit(X_train, y_train))
cv.best_params_
cv.best_score_
cv.best_estimator_

see_models()

#Train set score : 0.887230046948
#Test set score : 0.877062937063
#Wall time: 1min 46s

####  grid search is costly, you can also have a look at RandomizedSearchCV. More interesting, let's try genetic algorithm
# how they work :http://www.theprojectspot.com/tutorial-post/creating-a-genetic-algorithm-for-beginners/3

paramgrid = { # much bigger paramgrid than before => with a full grid search, 4*3*41*3*17*4 = 100 368 models would be tried
        "activation" : ["identity", "logistic","tanh","relu"],
        "solver": ["lbfgs","sgd","adam"],
        "alpha" : [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        "learning_rate":["constant","invscaling","adaptive"],
        "learning_rate_init": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500],
        "hidden_layer_sizes": [(100,80,50),(100,100,100,100,100),(100,100,100),(50,50,50)]
        }

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator = MLPClassifier(),
                                   params = paramgrid, 
                                   scoring = "accuracy",
                                   cv=StratifiedKFold(n_splits = 4),
                                   verbose = 1,
                                   population_size = 10,
                                   gene_mutation_prob = 0.10,
                                   gene_crossover_prob = 0.5,
                                   tournament_size = 3,
                                   generations_number = 5
                                   )
%time cv.fit(X_train, y_train)

#Best individual is: {'activation': 'relu', 'solver': 'sgd', 'alpha': 1, 'learning_rate': 'adaptive', 'learning_rate_init': 0.0001, 'hidden_layer_sizes': (50, 50, 50)}
#with fitness: 0.9295774647887324
#Wall time: 47.2 s



#### feature selection : recursive feature elimination
from sklearn.svm import SVC
classifier = SVC(kernel = "linear")

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator = classifier, 
              step = 1, # if >1, nb of features to remove at each iteration. If <1, percent of features to remove at each iteration
              cv = StratifiedKFold(n_splits = 2),
              scoring = 'accuracy',
              verbose = 1
              )

rfecv.fit(donnee.loc[:, donnee.columns != "target"], donnee.target)
rfecv.n_features_

# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#### feature selection + grid search
from sklearn.svm import SVC
classifier = SVC(kernel = "linear")

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV # recursive feature elimination cross validation
rfecv = RFECV(estimator = classifier, 
              cv = StratifiedKFold(n_splits = 2),
              scoring = 'accuracy',
              verbose = 1
              )

paramgrid = { 
        "estimator__C" : [0.1, 0.5, 1],
#        "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "estimator__shrinking": [True, False],
        } 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV #also works with RandomizedSearchCv and EvolutionaryAlgorithmSearchCV
cv = GridSearchCV(estimator = rfecv, 
                   param_grid = paramgrid, 
                   cv = StratifiedKFold(n_splits=4),
                   verbose = 1
                   )
%time models.append(cv.fit(X_train, y_train))
cv.best_params_
cv.best_score_
cv.best_estimator_
cv.best_estimator_.n_features_

see_models()



# see also sklearn.pipeline.Pipeline
