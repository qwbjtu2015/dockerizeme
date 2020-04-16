from os.path import dirname, join

from numpy import where, shape
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN, MiniBatchKMeans, SpectralClustering, KMeans
from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


class Data:
    def __init__(self):
        self.base_path = dirname(__file__)
        self.symbol = "CrudeOIL"
        self.tf = '60'
        self.mask = int(len(self.get_data().index)*0.6)
        self.train_data = self.get_data().iloc[:self.mask, :].pct_change().dropna()
        self.test_data = self.get_data().iloc[self.mask+1:, :].pct_change().dropna()
        
        self.train_features = scale(self.train_data.shift().dropna())
        self.test_features = scale(self.test_data.shift().dropna())
        self.train_targets = where(self.train_data.Close > 0, 1, 0)[1:]
        self.test_targets = where(self.test_data.Close > 0, 1, 0)[1:]

    def get_data(self):
        test_data = read_csv(filepath_or_buffer=join(self.base_path, 'data', '{0}{1}.csv'.format(self.symbol, self.tf)), names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], index_col='Date_Time', parse_dates=[[0, 1]])
        return test_data

    def accuracy(self, p):
        accuracy = accuracy_score(y_true=self.test_targets, y_pred=p)
        print("Accuracy {}".format(accuracy))

    def log_loss(self, p):
        logloss = log_loss(y_true=self.train_targets, y_pred=p)
        print("Log loss {}".format(logloss))

    def score(self, clf):
        score = clf.score(self.test_features, self.test_targets)
        print("Score: {}".format(score))
    
    def rf(self):
        rf = RandomForestClassifier(n_estimators=3, criterion='gini', max_depth=3, 
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_features='auto', max_leaf_nodes=5, min_impurity_split=1e-07, 
            bootstrap=True, oob_score=False, n_jobs=1, random_state=3, verbose=0, 
            warm_start=False, class_weight=None)
        model = rf.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)
        #~0.523
    
    def gbm(self):
        gbm = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
            n_estimators=10, subsample=1.0, criterion='friedman_mse', 
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_depth=3, min_impurity_split=1e-07, init=None, random_state=3, 
            max_features=None, verbose=0, max_leaf_nodes=5, warm_start=False, 
            presort='auto')
        model = gbm.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)
        #0.525
    
    def bc(self):
        bc = BaggingClassifier(base_estimator=None, n_estimators=10, 
            max_samples=1.0, max_features=0.5, bootstrap=True, 
            bootstrap_features=False, oob_score=False, warm_start=False, 
            n_jobs=1, random_state=None, verbose=0)
        model = bc.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model) 
        #0.502
    
    def kminibatch(self):
        km = MiniBatchKMeans(n_clusters=10, init='k-means++', max_iter=1000, 
            batch_size=100, verbose=0, compute_labels=True, random_state=None, 
            tol=0.0, max_no_improvement=10, init_size=None, n_init=3, 
            reassignment_ratio=0.01)
        model = km.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)  #xuinia kazkokia
    
    def kmeans(self):
        km = KMeans()
    
    def vc(self):
        vc = VotingClassifier(estimators=[('gbm', self.gbm()), ('rf', self.rf())], voting='hard', weights=None, n_jobs=1)
        model = vc.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)  #doesn't work foer defined
    
    def gpc(self):
        gpc = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', 
            n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, 
            random_state=None, multi_class='one_vs_rest', n_jobs=1)
        model = gpc.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)
        #0.5137
    
    def mlp(self):
        mlp = MLPClassifier(hidden_layer_sizes=(2000, 1000, 500, 300, 100, 50, 10), activation='identity', solver='adam', 
            alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
            power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001, 
            verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
            epsilon=1e-08)
        model = mlp.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)        
        #H1 - SP500 0.513
        #D1 - SP500 0.513
        #H1 - EURUSD 0.5188
        #H1 - Crude0.509
    
    def knc(self):
        knc = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', 
            leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
        model = knc.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model) 
        #0.51
    
    def dtc(self):
        dtc = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
            min_samples_split=2, min_samples_leaf=10, min_weight_fraction_leaf=0.0, 
            max_features=None, random_state=None, max_leaf_nodes=None, 
            min_impurity_split=1e-07, class_weight=None, presort=False)
        model = dtc.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)         
        #0.5
    
    def gnb(self):
        gnb = GaussianNB(priors=None)
        model = gnb.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)    
        #0.506
    
    def abc(self):
        abc = AdaBoostClassifier()
        model = abc.fit(X=self.train_features, y=self.train_targets)        
        self.score(clf=model)            
        #0.495


class PredictVolatility:
    def __init__(self):
        d = Data()
        self.train_features = d.train_features
        self.test_features = d.test_features
        self.train_targets = d.train_targets
        self.test_targets = d.test_targets
    
    def predict(self):
        print()
    
    
def main():
    data = Data()    
    data.mlp()
        
main()