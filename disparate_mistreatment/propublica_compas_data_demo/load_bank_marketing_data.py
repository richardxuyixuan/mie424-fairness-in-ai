from __future__ import division
from urllib.request import urlopen
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

sys.path.insert(0, '../disparate_mistreatment/') # the code for fair classification is in this directory
import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)

"""
    The bank marketing dataset can be obtained from: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
    The code will look for the data file in the present directory, after it is manually downloaded, with the csv file moved to the current directory.
"""

def load_bank_marketing_data():

	FEATURES_CLASSIFICATION = ["age","job","marital","education","default","housing","loan", "contact","month","day_of_week","poutcome"] #features to be used for classification
	CONT_VARIABLES = [] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CLASS_FEATURE = "y" # the decision variable
	SENSITIVE_ATTRS = ["age"]


	INPUT_FILE = "bank-additional-full.csv"

	# load the data and get some stats
	df = pd.read_csv(INPUT_FILE, sep = ";")
	df = df.dropna(subset=FEATURES_CLASSIFICATION) # dropping missing vals

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])

	""" Filtering the data """

	# data downloaded are already pro-processed

	""" Feature normalization and one hot encoding """

	y = data[CLASS_FEATURE]
	y[y=="yes"] = 1
	y[y=="no"] = -1
	y = y.astype('int32')

	# convert class label 'age' to a binary value where privileged is `age >= 25` and unprivileged is `age < 25` 
	for i in range(len(data["age"])):
		if int(data["age"][i]) >= 25:
			data["age"][i] = "privileged"
		elif int(data["age"][i]) < 25:
			data["age"][i] = "unprivileged"
	
	print("\nNumber of clients subscribed to a term deposit")
	print(pd.Series(y).value_counts())
	print("\n")

	X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
	x_control = defaultdict(list)

	feature_names = []
	for attr in FEATURES_CLASSIFICATION:
		vals = data[attr]
		if attr in CONT_VARIABLES:
			vals = [float(v) for v in vals]
			vals = preprocessing.scale(vals) # 0 mean and 1 variance  
			vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

		else: # for binary categorical variables, the label binarizer uses just one var instead of two
			lb = preprocessing.LabelBinarizer()
			lb.fit(vals)
			vals = lb.transform(vals)

		# add to sensitive features dict
		if attr in SENSITIVE_ATTRS:
			x_control[attr] = vals


		# add to learnable features
		X = np.hstack((X, vals))

		if attr in CONT_VARIABLES: # continuous feature, just append the name
			feature_names.append(attr)
		else: # categorical features
			if vals.shape[1] == 1: # binary features that passed through lib binarizer
				feature_names.append(attr)
			else:
				for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
					feature_names.append(attr + "_" + str(k))


	# convert the sensitive feature to 1-d array
	x_control = dict(x_control)
	for k in x_control.keys():
		assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
		x_control[k] = np.array(x_control[k]).flatten()

	# sys.exit(1)

	"""permute the date randomly"""
	perm = list(range(0,X.shape[0]))
	shuffle(perm)
	X = X[perm]
	y = y[perm]
	for k in x_control.keys():
		x_control[k] = x_control[k][perm]


	X = ut.add_intercept(X)

	feature_names = ["intercept"] + feature_names
	assert(len(feature_names) == X.shape[1])
	print("Features we will be using for classification are:"+str(feature_names)+"\n")


	return X, y, x_control
