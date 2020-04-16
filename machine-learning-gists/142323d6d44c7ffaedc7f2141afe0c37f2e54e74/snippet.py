from sklearn import tree
features = [[130, 6],[140, 9],[150, 90],[170, 777], [900,555]]
labels = [[0, 0], [0, 0], [1, 0], [1, 1], [2, 2]]
classifier = tree.DecisionTreeClassifier()
colorDictionary = {0: 'roja', 1:'verde', 2:'azul'}
typeDictionary = {0: 'naranja', 1: 'manzana', 2: 'pera'}
trainedClassifier = classifier.fit(features, labels)
prediction = trainedClassifier.predict([[100,9999]]);
print('Type Prediction: ', typeDictionary[prediction[0][0]], ' Color Prediction: ', colorDictionary[prediction[0][1]])
