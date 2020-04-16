import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.neighbors import NearestNeighbors
import math
import random


#Read in data
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#Column names
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
#Assign column names
df.columns = cols

#Scatter plot
x = df['sepal_length']
y = df['sepal_width']

plt.scatter(x, y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

#pick random point
random.seed()
pt = df.iloc[random.choice(df.index.tolist())]
pt['sepal_length']

#determine distances from random point
def dist_from_pt(p):
    return math.sqrt(((pt.sepal_length - p.sepal_length) ** 2) + ((pt.sepal_width - p.sepal_width) ** 2))

#set distances as values in new column
df['dist_from_pt'] = df[['sepal_length', 'sepal_width']].apply(func=dist_from_pt, axis=1)

#sort values by distance column
df_sorted = df.sort_values(by='dist_from_pt', ascending=True)

#define knn function
def knn(k):
	return df_sorted['class'][0:k].value_counts().index[0]

#print majority class based on number of neighbors inputted for k
print(knn(25))