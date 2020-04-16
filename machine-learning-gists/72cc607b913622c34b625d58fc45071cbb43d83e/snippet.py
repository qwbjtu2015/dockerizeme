__author__ = 'gavinwhyte'
import urllib2
import numpy
import random
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5: # labels that are 1.0 (positive examples)
            if predicted[i] > threshold:
                tp += 1.0 #correctly predicted positive
            else:
             fn += 1.0 # incorrectly predicted negative
        else: #labels that are 0.0 (negative example)
             if predicted[i] < threshold:
                 tn += 1.0
             else:
                 fp += 1.0
    rtn = [tp, fn, fp, tn]
    return rtn

# read in the rocks versus mines data set from uci.edu data repository

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = urllib2.urlopen(target_url)

xList=[]
labels =[]

for line in data:
    #split on comma
    row = line.strip().split(",")
    # print row
    #assign label 1.0 for "M" and 0.0 for "R"
    if (row[-1] == "M"):
        labels.append(1.0)
    else:
        labels.append(0.0)
        #remove label from row
    row.pop()
    #convert row to floats
    floatrow = [float(num) for num in row]
    xList.append(floatrow)

# divide attribute matrix and label vector into training (2/3 of data)

# and test sets (1/3 of data)

indices = range(len(xList))


print(indices)

xListTest = [xList[i] for i in indices if i%3 == 0 ]

xListTrain = [xList[i] for i in indices if i%3 != 0]

labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]


#from list of imput into numpy arrays to match input class
#for scikit-learn linear model
xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)

xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

#check the shapes to see what they look like
print("Shape of xTrain array", xTrain.shape)
print("Shape of yTrain array", yTrain.shape)
print("Shape of xTest array", xTest.shape)
print("Shape of yTest array", yTest.shape)

#train linear regression model

rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain,yTrain)

#generate predictions on in-sample error
trainingPredictions = rocksVMinesModel.predict(xTrain)
#print("Some values predicted by model",
#      trainingPredictions[0:5],
#      trainingPredictions[-6:-1])

print (trainingPredictions)


print("Some values predicted by model",
      trainingPredictions[0:12],
      trainingPredictions[-7:-1])

#generate confusion matrix for predictions on training set (in sample data)
confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)

#pick the threshold value and generate confusion matrix entries
tp = confusionMatTrain[0]; fn = confusionMatTrain[1]
fp = confusionMatTrain[2]; tn = confusionMatTrain[3]

print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " +
      str(fp) + "\ttn = " + str(tn) + '\n' )

#generate predicitions on out-sample data
testPredictions = rocksVMinesModel.predict(xTest)

#generate confusion matrix from predicitions on out-of-sample data
conMatTest = confusionMatrix(testPredictions, yTest, 0.5)

#pick threshold value and generate confusion matrix entries

tp = conMatTest[0]; fn = conMatTest[1]
fp = conMatTest[2]; tn = conMatTest[3]

print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " +
      str(fp) + "\ttn = " + str(tn) + '\n')


#generate ROC curve for in-sample

fpr, tpr, thresholds = roc_curve(yTrain,trainingPredictions)
roc_auc = auc(fpr, tpr)
print('Auc for in-sample ROC Curve: %f' % roc_auc)

# Plot ROC Curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve(area = %0.2f' % roc_auc)
pl.plot ([0,1], [0,1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False positive Rate')
pl.ylabel('True Positive Rate')
pl.title('In sample ROC rocks versus mines')
pl.legend(loc="lower right")
pl.show()

#generate ROC curve for Out-of-sample

fpr, tpr, thresholds = roc_curve(yTest,testPredictions)
roc_auc = auc(fpr, tpr)
print('Auc for in-sample ROC Curve: %f' % roc_auc)

# Plot ROC Curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve(area = %0.2f' % roc_auc)
pl.plot ([0,1], [0,1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Out-of-sample ROC rocks versus mines')
pl.legend(loc="lower right")
pl.show()
