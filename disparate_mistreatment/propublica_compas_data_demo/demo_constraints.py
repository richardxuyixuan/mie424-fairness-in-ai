import os,sys
import numpy as np
from load_compas_data import *
from load_bank_marketing_data import *
from load_meps_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import funcs_disp_mist as fdm

def test_data(dataset):
	
	""" Generate the synthetic data """
	data_type = 1
	if dataset == "compas":
		X, y, x_control = load_compas_data()
		sensitive_attr = "race"
	elif dataset == "bank_marketing":
		X, y, x_control = load_bank_marketing_data()
		sensitive_attr = "age"
	elif dataset == "meps":
		X, y, x_control = load_meps_data()
		sensitive_attr = "RACE"
	sensitive_attrs = list(x_control.keys())
	
	""" Split the data into train and test """
	train_fold_size = 0.5
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	cons_params = None # constraint parameters, will use them later
	loss_function = "logreg" # perform the experiments with logistic regression
	EPS = 1e-5

	def train_test_classifier():
		w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)

		train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)

		
		# accuracy and FPR are for the test because we need of for plotting
		return w, test_score, s_attr_to_fp_fn_test
		

	""" Classify the data while optimizing for accuracy """
	print("== Unconstrained (original) classifier ==")
	w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons = train_test_classifier()
	print("\n-----------------------------------------------------------------------------------\n")

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	print("\n\n== Constraints on FPR ==")	# setting parameter for constraints
	cons_type = 1 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {sensitive_attr: {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons  = train_test_classifier()
	print("\n-----------------------------------------------------------------------------------\n")

	print("\n\n== Constraints on FNR ==")	# setting parameter for constraints
	cons_type = 2 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {sensitive_attr: {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons  = train_test_classifier()
	print("\n-----------------------------------------------------------------------------------\n")
	print("\n\n== Constraints on both FPR and FNR ==")	# setting parameter for constraints
	cons_type = 4 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {sensitive_attr: {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons  = train_test_classifier()
	print("\n-----------------------------------------------------------------------------------\n")

	return


def main():
	data = input("Enter bank_marketing, meps, or compas to run Algorithm 2 for the specific dataset: ")
	test_data(data)

if __name__ == '__main__':
	main()