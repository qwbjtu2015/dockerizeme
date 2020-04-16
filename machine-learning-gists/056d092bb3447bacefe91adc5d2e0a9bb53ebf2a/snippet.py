# Code example for:
# Hello World - Machine Learning Recipes #1 - Google Developers
# https://www.youtube.com/watch?v=cKxRvEZd3Mw

from sklearn import tree

# Bumpy = 0, Smooth = 1
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Apple = 0, Orange = 1
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([[160, 0]])
