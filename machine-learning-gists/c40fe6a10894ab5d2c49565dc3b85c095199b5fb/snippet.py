#!/usr/bin/python
"""
Run python checkerboards.py

Example from:
M. Hein (2009). Binary Classification under Sample Selection Bias, In Dataset Shift in Machine Learning, chap. 3, pp. 41-64. The MIT Press.
"""

from __future__ import division
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import Tkinter as Tk
import sys
import numpy as np
from itertools import izip
from functools import partial

from scikits.learn.svm import SVC


def generate_data(sample_size=200, pd=[[0.4,0.4],[0.1,0.1]]):
    pd = np.array(pd)
    pd /= pd.sum()
    offset = 50
    bins = np.r_[np.zeros((1,)),np.cumsum(pd)]
    bin_counts = np.histogram(np.random.rand(sample_size), bins)[0]
    data = np.empty((0,2))
    targets = []
    for ((i,j),p),count in zip(np.ndenumerate(pd),bin_counts):
        xs = np.random.uniform(low=0.0,high=50.0, size=count) + j*offset
        ys = np.random.uniform(low=0.0,high=50.0, size=count) + -i*offset
        data = np.vstack((data,np.c_[xs,ys]))
        if i == j:
            targets.extend([1]*count)
        else:
            targets.extend([-1]*count)
    return np.c_[data,targets]

class Model(object):
    def __init__(self):
        self.observers = []
        self.trainerr = "-"
        self.testerr = "-"
        self.surface = None

    def changed(self):
        for observer in self.observers:
            observer.update(self)

    def set_train(self,data):
        self.train = data

    def set_test(self,data):
        self.test = data

    def add_observer(self,observer):
        self.observers.append(observer)

    def set_testerr(self, testerr):
        self.testerr = testerr

    def set_trainerr(self, trainerr):
        self.trainerr = trainerr

    def set_surface(self,surface):
        self.surface = surface

class Controller(object):
    def __init__(self, model):
        self.model = model

    def generate_data(self):
        print "generate data called"
        self.model.set_train(generate_data(pd=self.train_pd.get_pd()))
        self.model.set_test(generate_data(pd=self.test_pd.get_pd()))
        self.model.set_surface(None)
        self.model.set_testerr("-")
        self.model.set_trainerr("-")
        self.model.changed()

    def classify(self, kernel="linear"):
        print "classifying data"
        train = self.model.train
        
        samples = train[:,:2]
        labels = train[:,2].ravel()
        accs = []
        cs = 2.0**np.arange(-5,4,2)
        gammas = [0.0] if kernel == "linear" else 2.0**np.arange(-15,3,2)

        clf = SVC(kernel=kernel, C=1, probability=True)
        clf.fit(samples, labels)
        
        print "--------------------------------------------------"
        #print "Accuracy=%f\tC=%f\tgamma=%f\t" % accs[0]
        print "--------------------------------------------------"
        
        
        train_err = 1.0 - clf.score(samples,
                              labels)
        test_err = 1.0 - clf.score(self.model.test[:,:2],
                             self.model.test[:,2].ravel())
        X1, X2, Z = self.decision_surface(clf)
        self.model.set_trainerr("%.2f" % train_err)
        self.model.set_testerr("%.2f" % test_err)
        self.model.set_surface((X1, X2, Z))
        self.model.changed()

    def decision_surface(self, clf):
        delta = 0.25
        x = np.arange(0.0, 100.1,  delta)
        y = np.arange(-50.0, 50.1, delta)
        X1, X2 = np.meshgrid(x, y)
        Z = np.empty(X1.shape)
        for (i,j),val in np.ndenumerate(X1):
            x1 = val
            x2 = X2[i,j]
            p = clf.predict_proba([x1, x2])
            Z[i, j] = p[0,1] # prob of pos class
        return X1, X2, Z        

    def quit(self):
        sys.exit()

    def set_train_pd(self, train_pd):
        self.train_pd = train_pd
        
    def set_test_pd(self, test_pd):
        self.test_pd = test_pd


class View(object):
    def __init__(self,root):
        f = Figure(figsize=(10,5), dpi=100)
        train_plot = f.add_subplot(121)
        train_plot.set_title("Training Distribution")
        test_plot = f.add_subplot(122)
        test_plot.set_title("Test Distribution")
        train_plot.set_xticks([])
        test_plot.set_yticks([])
        train_plot.set_yticks([])
        test_plot.set_xticks([])
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, root )
        toolbar.update()
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.f = f
        self.test_plot = test_plot
        self.train_plot = train_plot
        self.toolbar = toolbar
        self.canvas = canvas
        self.hascolormaps = False
        self.trainerr_text = self.f.text(0.2, 0.05, "Errorrate = -")
        self.testerr_text = self.f.text(0.6, 0.05, "Errorrate = -")

    def update(self,model):
        self.train_plot.clear()
        self.test_plot.clear()
        self.plot_data(self.train_plot,model.train,
                       title="Training Distribution")
        self.plot_data(self.test_plot,model.test,
                       title="Test Distribution")
        print "training error rate: ", model.trainerr
        print "test error rate: ", model.testerr
        self.plot_errors(model.trainerr,model.testerr)

        if model.surface != None:
            CS = self.plot_decision_surface(self.train_plot,model.surface)
            CS = self.plot_decision_surface(self.test_plot,model.surface)
            self.plot_colormaps(CS)

        self.canvas.show()

    def plot_data(self, fig, data, title = ""):
        pos_data = data[data[:,2]==1]
        neg_data = data[data[:,2]==-1]
        fig.plot(pos_data[:,0], pos_data[:,1], 'wo',neg_data[:,0],
                 neg_data[:,1], 'ko')
        fig.set_ylim((-50,50))
        fig.set_xlim((0,100))
        fig.set_xticks([])
        fig.set_yticks([])
        fig.set_title(title)

    def plot_decision_surface(self, fig, surface):
        X1, X2, Z = surface
        levels = np.arange(0.0, 1.1, 0.1)
        CS = fig.contourf(X1, X2, Z, levels,
                        cmap=matplotlib.cm.bone,
                        origin='lower')
        return CS
        
    def plot_colormaps(self, CS):
        if not self.hascolormaps:
            self.f.colorbar(CS,ax = self.train_plot)
            self.f.colorbar(CS,ax = self.test_plot)
            self.hascolormaps = True

    def plot_errors(self, trainerr, testerr):
        self.trainerr_text.set_text("Errorrate = %s" % trainerr)
        self.testerr_text.set_text("Errorrate = %s" % testerr)


class Table(object):
    def __init__(self, pd, *args, **kargs):
        master = Tk.Frame(*args, **kargs)
        self.master = master
        
        self.e1 = Tk.Entry(master,width=5)
        self.e1.insert(0, pd[0,0])
        self.e2 = Tk.Entry(master,width=5)
        self.e2.insert(0, pd[0,1])
        self.e3 = Tk.Entry(master,width=5)
        self.e3.insert(0, pd[1,0])
        self.e4 = Tk.Entry(master,width=5)
        self.e4.insert(0, pd[1,1])
        self.e1.grid(row=0, column=0)
        self.e2.grid(row=0, column=1)
        self.e3.grid(row=1, column=0)
        self.e4.grid(row=1, column=1)

    def get_pd(self):
        return [[float(self.e1.get()), float(self.e2.get())],
                 [float(self.e3.get()), float(self.e4.get())]]

    def pack(self,**kargs):
        self.master.pack(**kargs)

    def grid(self,**kargs):
        self.master.grid(**kargs)

def learnModel(train):
    pass
    

def main(argv):
    root = Tk.Tk()
    root.wm_title("Checkerboards")
    view = View(root)
    model = Model()
    model.add_observer(view)
    controller = Controller(model)
    train_label = Tk.Label(root, text="Train Marginal Distribution:")
    train_label.pack(side=Tk.LEFT)
    train_pd = Table(np.array([[0.4,0.4],[0.1,0.1]]), root,
                     width=100, height=100)
    train_pd.pack(side=Tk.LEFT)
    test_label = Tk.Label(root, text="Test Marginal Distribution:")
    test_label.pack(side=Tk.LEFT)
    test_pd = Table(np.array([[0.4,0.1],[0.4,0.1]]), root,
                    width=100, height=100)
    test_pd.pack(side=Tk.LEFT)

    controller.set_train_pd(train_pd)
    controller.set_test_pd(test_pd)

    generate_button = Tk.Button(master=root,
                                text='Generate Data',
                                command=controller.generate_data)
    generate_button.pack(side = Tk.LEFT)

    svm_linear_button = Tk.Button(master=root,
                                text='Classify LINEAR',
                                command=partial(controller.classify,
                                                kernel="linear"))
    svm_linear_button.pack()

    svm_rbf_button = Tk.Button(master=root,
                                text='Classify RBF',
                                command=partial(controller.classify,
                                                kernel="rbf"))
    svm_rbf_button.pack()
    Tk.mainloop()

if __name__ == "__main__":
    main(sys.argv)
