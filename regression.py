import theano
from theano import tensor as T
import numpy as np
from scipy.cluster.vq import whiten
import sys
import math as m
import itertools
from sklearn.cross_validation import train_test_split
import random as r

INITIAL_WEIGHT_MAX = .001
LEARNING_RATE = 0.00001

def parse(data_file_name, predict_index, ignore_indices, **options):
	data_file = open(data_file_name, 'r')
	lines = data_file.read().splitlines()
	x = []
	y = []
	for i, line in enumerate(lines):
		if i == 0 or i == 1:
			continue
		datas = line.split()
		x_category = []
		for i, data in enumerate(datas):
			if ignore_indices.has_key(i):
				continue
			if i == predict_index:
				if data == 'T':
					y.append(1.0)
				elif data == 'F':
					y.append(0.0)
				else:
					y.append(float(data))
				continue
			x_category.append(float(data))
		x.append(x_category)
	x = whiten(np.array(x)) if options.get('whiten_x') else np.array(x)
	y = whiten(np.array(y)) if options.get('whiten_y') else np.array(y)
	x = x - x.mean() if options.get('mean_center_x') else x
	y = y - y.mean() if options.get('mean_center_y') else y
	return (x, y)


def run_regression(trials, x, y, **options):
	x_dim = len(x[0])
	weight_vec = theano.shared(np.random.randn(x_dim, 1) * INITIAL_WEIGHT_MAX)
	symbolic_x = T.fmatrix('x')
	symbolic_y = T.fvector('y')
	output = T.dot(symbolic_x, weight_vec).transpose()
	cost = T.sum(T.pow(output - symbolic_y, 2))
	weight_grad = T.grad(cost, wrt=weight_vec)
	updates = [[weight_vec, weight_vec - weight_grad * LEARNING_RATE]]
	train_f = theano.function(inputs=[symbolic_x, symbolic_y], outputs=cost, updates=updates, allow_input_downcast=True)
	test_f = theano.function(inputs=[symbolic_x, symbolic_y], outputs=cost, allow_input_downcast=True)
	output_f = theano.function(inputs=[symbolic_x], outputs=output, allow_input_downcast=True)
	trial_cost = 0.0
	for i in range(trials):
		trial_cost = train_f(x, y)
		if options.get('verbose'):
			print 'estimated y: '
			print output_f(x)
			print 'true y'
			print y
			print 'sum of squared errors:'
			print trial_cost
	if options.get('x_test') is not None and options.get('y_test') is not None:
		return test_f(options.get('x_test'), options.get('y_test')) / len(options.get('y_test'))
	return trial_cost


def mallow_cp(n, features, SSE_reduced, MSE_full):
	# formulation taken from http://www.statistics4u.info/fundstat_eng/cc_varsel_mallowscp.html
	return (SSE_reduced / MSE_full) - n + 2. * features


def aic(n, features, SSE):
	return -2. * m.log(SSE / n) + 2. * features


def cross_validation(x, y, folds, trials=5000):
	# test size is the percentage of the data that should be in the test set
	test_size = float(len(y)) / float(folds) / float(len(y))
	test_SSE = 0.0
	for i in range(folds):
		x_train, x_test, y_train, y_test = train_test_split(
			x, y, test_size=test_size)
		test_SSE += (1.0 / folds) * run_regression(trials, x_train, y_train, x_test=x_test, y_test=y_test)
	return test_SSE


def best_subset_selection(x, y, trials=5000):
	x_dim = len(x[0])
	y_dim = len(y)
	best_subset_SSE = [sys.float_info.max for dim in range(x_dim)]
	best_subset_columns = [[] for dim in range(x_dim)]
	for num_features in range(1, x_dim + 10):
		for combination in itertools.combinations(range(x_dim), num_features):
			print 'testing column combination: ' + str(combination)
			subset_x = x[:, list(combination)]
			subset_SSE = run_regression(trials, subset_x, y)
			if subset_SSE < best_subset_SSE[num_features - 1]:
				best_subset_SSE[num_features - 1] = subset_SSE
				best_subset_columns[num_features - 1] = list(combination)
	print 'best subset columns are: ' + str(best_subset_columns)
	print 'corresponding best SSE is: ' + str(best_subset_SSE)
	for i, subset in enumerate(best_subset_columns):
		ten_fold_error = cross_validation(x[:, subset], y, 10)
		five_fold_error = cross_validation(x[:, subset], y, 5)
		aic_score = aic(y_dim, i + i, best_subset_SSE[i])
		print 'Subset consists of columns: ' + str(subset)
		print 'Train MSE was ' + str(best_subset_SSE[i] / y_dim)
		print 'Test MSE is (10 fold cross validation): ' + str(ten_fold_error)
		print 'Test SSE is (5 fold cross validation): ' + str(five_fold_error)
		print 'AIC is ' + str(aic_score)


def forward_selection(x, y, trials=5000):
	x_dim = len(x[0])
	y_dim = len(y)
	# calculate the MSE with all features
	MSE_full = run_regression(trials, x, y) / float(y_dim)
	# we don't have a mallow_cp previous since we are starting with only one feature
	# initialize the previous mallow_cp to infinity
	mallow_cp_previous = sys.float_info.max
	# all the features we can select from
	potential_columns = set(range(x_dim))
	# the features we have selected for the model so far
	used_columns = set([])
	# start out with one feature, iterate up to the full number of features
	# or until the mallow_cp starts increasing
	for num_features in range(1, x_dim + 1):
		current_columns = list(used_columns)
		# the best SSE for this number of features
		best_SSE_reduced = sys.float_info.max
		# the next best feature (column) to add to the model out of the remaining available features
		best_next_column = -1
		# iterate through all remaining features, select the one which produces the best SSE
		for column in (potential_columns - used_columns):
			reduced_x = x[:, current_columns + [column]]
			SSE_reduced = run_regression(trials, reduced_x, y)
			if SSE_reduced < best_SSE_reduced:
				best_SSE_reduced = SSE_reduced
				best_next_column = column
		# compare the mallow_cp of the model + this new feature to that of the model before the feature was added
		if mallow_cp(y_dim, num_features, best_SSE_reduced, MSE_full) < mallow_cp_previous:
			used_columns.add(best_next_column)
			mallow_cp_previous = mallow_cp(y_dim, x_dim, best_SSE_reduced, MSE_full)
		else: # if the mallow_cp increases, we break
			break
	print str(list(used_columns)) + ' are the best columns, mallow_cp is ' + str(mallow_cp_previous)


def backward_selection(x, y, trials=5000):
	x_dim = len(x[0])
	y_dim = len(y)
	# get the MSE for all the trials
	MSE_full = run_regression(trials, x, y) / y_dim
	# calculate the mallow for this
	mallow_cp_previous = mallow_cp(y_dim, x_dim, y_dim * MSE_full, MSE_full)
	# initially, we use all columns for backward selection
	used_columns = range(x_dim)
	# start with the max number of features and work backward
	for num_features in range(x_dim, 1, -1):
		# the best SSE for this number of features
		best_SSE_reduced = sys.float_info.max
		# best column to remove this iteration
		best_column_to_remove = 0
		# try out removing all columns
		for column_to_exclude in range(num_features):
			columns_minus_one = []
			for i in range(len(used_columns)):
				columns_minus_one.append(used_columns[i]) if i != column_to_exclude else None
			reduced_x = x[:, columns_minus_one]
			# calculate the SSE for the reduced number of columns
			SSE_reduced = run_regression(trials, reduced_x, y)
			if SSE_reduced < best_SSE_reduced:
				best_SSE_reduced = SSE_reduced
				best_column_to_remove = column_to_exclude
		# see if the most successful set of columns for this number of features out performs the
		# previous best set of columns
		if mallow_cp(y_dim, num_features, best_SSE_reduced, MSE_full) < mallow_cp_previous:
			del(used_columns[best_column_to_remove])
			mallow_cp_previous = mallow_cp(y_dim, x_dim, best_SSE_reduced, MSE_full)
		else: # if the mallow cp increased when we removed the least helpful feature, we are done
			break
	print str(list(used_columns)) + ' are the best columns, mallow_cp is ' + str(mallow_cp_previous)


if __name__ == "__main__":
	# get the car data
	x_cars, y_cars = parse('data.txt', 5, {0:True},
		whiten_x=True, whiten_y=True, mean_center_x=True, mean_center_y=True)
	#run_regression(1000, x_cars, y_cars, verbose=True)
	backward_selection(x_cars, y_cars)
	forward_selection(x_cars, y_cars)
	# get the prostate data
	x, y = parse('prostateData.txt', predict_index=10, ignore_indices={0:True},
		whiten_x=True, whiten_y=False, mean_center_x=True, mean_center_y=False)
	#run_regression(5000, x, y)