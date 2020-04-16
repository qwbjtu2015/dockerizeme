import pandas as pd
df = pd.read_csv(r"D:\ecliips\UDEMY\datasets by UDEMY\data preprocessing\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv")
print(df.head())


#X = X.reshape(y.shape[0], 1)
# in case categorical data we need to convert text category into interger by encoding them
# we can achieve encoding by two methods:
    #1. LabelEncoder
    #2. OneHotEncoder
#which can be imported from preprocessor

#df = df.interpolate(method = "linear",axis = 0)
#print(df)

# X = df.iloc[:, :-1].values
# y = df.iloc[:, 3].values
#
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# le = LabelEncoder()
# X[:,0]= le.fit_transform(X[:,0])
# le1 = OneHotEncoder(categorical_features=[0])
# X = le1.fit_transform(X)
# print(X)


# label encoder ncode category to 0,1,2,3 etc.
# it is not suitable for multiple category data as model consider this category relation as greater than and less than
# i.e it will consider 2>1,1>0 in this manner it will classify

# therefore if there are more than 2 category encode data by OneHotEncode
# # we need to pass categorical_feature to it
# # to catogorical_feature we pass basically the index of categorical column  index i.e here index of country column=[0]r

# from sklearn.preprocessing import OneHotEncoder
# le2 = OneHotEncoder(categorical_features=[0])
# X = le2.fit_transform(X).toarray()
# print(X)

# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder(categorical_features = [-1])
# X = onehotencoder.fit_transform(X).toarray()


import numpy as np
X = df.iloc[:, :-1].values

from sklearn.impute import  SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]= imputer.transform(X[:,1:3])
#print(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features =[0]) # 0 th column need to encode therefore catogorical_feature = 0
X = onehotencoder.fit_transform(X).toarray()
print(X[:,0:3])

# print(X[:,0])
# print(X[:,1])
# print(X[:,2])

from sklearn.preprocessing import StandardScaler
SC= StandardScaler()
X = SC.fit_transform(X)
print(X[:,:-1],end= " ")
