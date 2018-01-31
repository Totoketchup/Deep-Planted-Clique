# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from data import get_data_adjacency, train_test_valid_shuffle
import argparse
import os
from itertools import product
# make results reproducible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from networks import Planted, Optimizer, Trainer, CNN_description, FF_description

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train Network on Planted CLique Data')
	# Add arguments
	parser.add_argument(
		'--data', help='dataset used', required=True)
	parser.add_argument(
		'--text', help='number of trials', action="store_true")
	parser.add_argument(
		'--data_path', help='dataset path', required=False, default='data/')
	parser.add_argument(
		'--search_grid', help='Do Search Grid', action="store_true")
	parser.add_argument(
		'--trials', type=int, help='number of trials', required=False, default=1)
	parser.add_argument(
		'--k', type=int, help='size of clique', required=False, default=10)
	parser.add_argument(
		'--binary', help='Unique output', action="store_true")
	parser.add_argument(
		'--train_ratio', type=float, help='ratio of train set', required=False, default=0.8)
	parser.add_argument(
		'--test_ratio', type=float, help='ratio of test set', required=False, default=0.1)
	parser.add_argument(
		'--valid_ratio', type=float, help='ratio of valid set', required=False, default=0.1)	
	parser.add_argument(
		'--nb_samples', type=int, help='Truncate the number of samples used', required=False, default=0)	
	parser.add_argument(
		'--save', help='Unique output', action="store_true")
	parser.add_argument(
		'--excluding', help='Excluding top k', action="store_true")
	parser.add_argument(
		'--import_test_data', help='Import custom test data', required=False, default='')	

	args = parser.parse_args()

	if args.binary:
		classes = 1
	else:
		classes = 2

	if not args.text:
		x_vals, y_vals = get_data_adjacency(args.data, args.data_path, one_hot = not args.binary)

	if args.nb_samples == 0:
		args.nb_samples = len(x_vals)

	input_dim = x_vals.shape[-1]

	trials = args.trials

	args.data = args.data.replace(":", "_")

	if args.search_grid:

		vals = train_test_valid_shuffle(x_vals, y_vals,
										train_ratio=args.train_ratio, 
										valid_ratio=args.valid_ratio, 
										test_ratio=args.test_ratio,
										nb_samples=args.nb_samples,
										import_test_data=args.import_test_data)
		x_vals_train, y_vals_train = vals[0]
		x_vals_test, y_vals_test = vals[1]
		x_vals_valid, y_vals_valid = vals[2]

		print 'Search Grid'

		h = [50, 100, 200, 500, 1000, 2000]
		layers = [2]
		hidden = []
		for l in layers:
			hidden += product(*(h,)*l)

		search_space = {
			'hidden' : hidden,
			'dropout' : [0.2, 0.3, 0.5, 0.8],
			'learning_rate' : [0.01, 0.001],
			'batch_size' : [512, 1024, 2048],
			'optimizer' : [tf.train.AdamOptimizer],
			'epochs' : [400],
			'classes' : [classes],
			'input_dim' : [input_dim],
			'activation' : [tf.nn.sigmoid],
			'data' : [args.data]
		}

		search_grid = Optimizer(search_space, Planted)
		search_grid.search(x_vals_train, y_vals_train, 
						x_vals_valid, y_vals_valid, 
						x_vals_test, y_vals_test)

	else:
		print 'Multiple trials runs'

		params = {
			'k' : args.k,
			'learning_rate' : 0.1,
			'batch_size' : 16,
			'optimizer' : tf.train.AdamOptimizer,
			'epochs' : 200,
			'classes' : classes,
			'input_dim' : input_dim,
			'data' : args.data,
			'activation' : tf.nn.sigmoid,
			'train_ratio': args.train_ratio,
			'import_test_data' : args.import_test_data,
			'nb_samples': args.nb_samples,
			'save': args.save,
			'excluding': args.excluding
		}

		trials = args.trials


		test_acc, valid_acc, accuracy = Trainer(params, Planted, trials).train(x_vals, y_vals)

		print 'On '+str(trials)+' Trials:'
		print 'Test mean = '+str(np.mean(test_acc))+'  std= '+str(np.std(test_acc))
		print 'Valid mean = '+str(np.mean(valid_acc))+'  std= '+str(np.std(valid_acc))
		print 'Accuracy mean = '+str(np.mean(accuracy))+'  std= '+str(np.std(accuracy))
