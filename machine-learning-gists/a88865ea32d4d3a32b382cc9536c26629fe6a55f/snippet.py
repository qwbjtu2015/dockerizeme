from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import shutil

import tempfile
import urllib

%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

# printing out the versions 
print(tf.__version__)
print(pd.__version__)

cwd = os.getcwd()
cwd

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

TRAIN_FILE_NAME = cwd + "/adult.data.csv"
TEST_FILE_NAME = cwd + "/adult.test.csv"


#Columns

CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", 
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]


df_train = pd.read_csv(TRAIN_FILE_NAME, names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(TEST_FILE_NAME, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)

df_test.head()

#Construct a new column named label as the ouput column

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


# Create an input function which converts the data to tensors/sparse tensors

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

# Engineering the features of the columns 

gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])

#Define the sparse categorical columns with hash_buckets when we dont know the number of unique variables 
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=1000)

# Define the base featured columns with continous values
# Below features are not used
'''age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
'''

# --------- ---------- Define the Simple logistic regression model  --------- ---------- #

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race],
  model_dir=model_dir)

# Training the model 
m.fit(input_fn=train_input_fn, steps=200)

# Evaluating the model 
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

### saving the sample model 
feature_columns = set([gender, native_country, education, occupation, workclass, marital_status, race])

#Save Model into saved_model.pbtxt file (possible to Load in Java)
tfrecord_serving_input_fn = tf.contrib.learn.build_parsing_serving_input_fn(tf.contrib.layers.create_feature_spec_for_parsing(feature_columns))  
m.export_savedmodel(export_dir_base="/Users/gagandeep.malhotra/Documents/SampleTF_projects/tempppp", serving_input_fn = tfrecord_serving_input_fn,as_text=False)

#Loading mode for prediction

from tensorflow.contrib import predictor
export_dir = "/Users/Documents/SampleTF_projects/tempppp/1510877466/"
predict_fn = predictor.from_saved_model(export_dir, signature_def_key=None)

input11 = df_train[2:3]

K_CATEGORICAL_COLUMNS = ["gender", "native_country", "education", "occupation", "workclass", "marital_status", "race"]

def test_ip(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  #continuous_cols = {k: tf.constant(df[k].values)
  #                   for k in K_CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in K_CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  #feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  return categorical_cols

# Make a dict to be passed to the predict function containg the test data 
dict_test = test_ip(input11)

predictions = predict_fn(dict_test)
print(predictions['probabilities'])


