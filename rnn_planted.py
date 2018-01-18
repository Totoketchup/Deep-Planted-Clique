import tensorflow as tf
from data import get_data, train_test_valid_shuffle, get_txt_data
import os
import numpy as np
from networks import RNN, Optimizer, Trainer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

# K=10 E=10 M = 2
#{'layers': 3, 'optimizer': <class 'tensorflow.python.training.adam.AdamOptimizer'>, 
# 'learning_rate': 0.001, 'batch_size': 2048, 'epochs': 70, 'valid_accuracy': 0.65100002, 'hidden': 30, 
# 'dropout': 0.6}
#Test Accuracy = 0.663


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train Network on Planted CLique Data')
	# Add arguments
	parser.add_argument(
		'--data', help='dataset used', required=True)
	parser.add_argument(
		'--text', help='Is it a .txt file ?', action="store_true")
	parser.add_argument(
		'--data_path', help='dataset path', required=False, default='data/')
	parser.add_argument(
		'--topological', help='Use topological features', action="store_true")
	parser.add_argument(
		'-td', '--trunc_dim', type=int, help='Truncate the size of feature dimension', required=False, default=0)
	parser.add_argument(
		'--search_grid', help='Do Search Grid', action="store_true")
	parser.add_argument(
		'--trials', type=int, help='number of trials', required=False, default=1)

	args = parser.parse_args()

	if not args.text:
		x_vals, y_vals = get_data(args.data, args.data_path, args.topological)
	else:
		x_vals, y_vals = get_txt_data(args.data, args.data_path)

	x_vals = np.squeeze(x_vals)
	input_dim = x_vals.shape[-1]

	trials = args.trials

	args.data = args.data.replace(":", "_")

	if args.search_grid:

		vals = train_test_valid_shuffle(x_vals, y_vals)
		x_vals_train, y_vals_train = vals[0]
		x_vals_test, y_vals_test = vals[1]
		x_vals_valid, y_vals_valid = vals[2]

		print 'Search Grid'

		search_space = {
			'hidden' : [30],
			'layers' : [3],
			'dropout' : [0.6],
			'learning_rate' : [0.001],
			'batch_size' : [2048],
			'optimizer' : [tf.train.AdamOptimizer],
			'epochs' : [200],
			'classes' : [2],
			'input_dim' : [input_dim]
		}

		search_grid = Optimizer(search_space, RNN)
		search_grid.search(x_vals_train, y_vals_train, 
						x_vals_valid, y_vals_valid, 
						x_vals_test, y_vals_test)

	else:
		print 'Mutliple trials runs'

		params = {
			'hidden' : 30,
			'layers' : 3,
			'dropout' : 0.6,
			'learning_rate' : 0.001,
			'batch_size' : 2048,
			'optimizer' : tf.train.AdamOptimizer,
			'epochs' : 10,
			'classes' : 2,
			'input_dim' : input_dim
		}

		trials = args.trials


		test_acc, valid_acc = Trainer(params, RNN, trials).train(x_vals, y_vals)

		print 'On '+str(trials)+' Trials:'
		print 'Test mean = '+str(np.mean(test_acc))+'  std= '+str(np.std(test_acc))
		print 'Valid mean = '+str(np.mean(valid_acc))+'  std= '+str(np.std(valid_acc))


