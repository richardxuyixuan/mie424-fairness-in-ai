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
    The Medical Expenditure Panel Survey dataset can be obtained from: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192
    To allow the dataset to be compatible with the codebase, download the "Data File, ASCII format", and convert it to CSV format.
	The code will look for the data file in the present directory, after the csv file is moved to the current directory.
"""
def race(row):
	if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
		result = 'White'
		return result
	else:
		result = 'Non-White'
		return result

def utilization(row):
    return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']


def load_meps_data():
	# TO DO: CHANGE THIS
	FEATURES_CLASSIFICATION = ['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PCS42',
                                 'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV', 'PERWT16F'] #features to be used for classification
	CONT_VARIABLES = ['AGE','PCS42','MCS42','K6SUM42', 'PERWT16F'] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CLASS_FEATURE = 'UTILIZATION' # the decision variable
	SENSITIVE_ATTRS = ['RACE']

	INPUT_FILE = "h192.csv"

	# load the data and get some stats
	df = pd.read_csv(INPUT_FILE)

	""" Filtering the data """
	df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
	df = df.rename(columns = {'RACEV2X' : 'RACE'})
	df = df[df['PANEL'] == 21]
	df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT16' : 'POVCAT', 'INSCOV16' : 'INSCOV'})
	df = df[df['REGION'] >= 0] # remove values -1
	df = df[df['AGE'] >= 0] # remove values -1
	df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9
	df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
	df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1
	df['TOTEXP16'] = df.apply(lambda row: utilization(row), axis=1)
	lessE = df['TOTEXP16'] < 10.0
	df.loc[lessE,'TOTEXP16'] = -1.0
	moreE = df['TOTEXP16'] >= 10.0
	df.loc[moreE,'TOTEXP16'] = 1.0
	
	df = df.rename(columns = {'TOTEXP16' : 'UTILIZATION'})
	df = df.dropna(subset=FEATURES_CLASSIFICATION) # dropping missing vals

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])

	""" Feature normalization and one hot encoding """

	y = data[CLASS_FEATURE]

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
