import tensorflow as tf
import numpy as np
from data import get_data, train_test_valid_shuffle,get_txt_data
import argparse
import os
# make results reproducible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from networks import CNN, Optimizer, Trainer, FF_description, CNN_description



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
	_, height, width = x_vals.shape
	x_vals = np.expand_dims(x_vals, 3)
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

		search_grid = Optimizer(search_space, CNN)
		search_grid.search(x_vals_train, y_vals_train, 
						x_vals_valid, y_vals_valid, 
						x_vals_test, y_vals_test)

	else:

		layers = [	
			CNN_description(16, [5,5], tf.nn.relu),
			CNN_description(32, [3,3], tf.nn.relu),
			CNN_description(64, [2,2], tf.nn.relu),
		]

		ffcs = [
			FF_description(256, tf.nn.relu),
			FF_description(128, tf.nn.relu),
			FF_description(2, None)
		]

		params = {
			'layers' : layers,
			'ffcs' : ffcs,
			'shape' : (height, width),
			'batch_norm' : True,
			'dropout' : 0.5,
			'learning_rate' : 0.001,
			'batch_size' : 512,
			'optimizer' : tf.train.AdamOptimizer,
			'epochs' : 10,
			'classes' : 2,
			'input_dim' : input_dim,
			'data' : args.data
		}

		trials = args.trials

		print x_vals.shape
		test_acc, valid_acc = Trainer(params, CNN, trials).train(x_vals, y_vals)

		print 'On '+str(trials)+' Trials:'
		print 'Test mean = '+str(np.mean(test_acc))+'  std= '+str(np.std(test_acc))
		print 'Valid mean = '+str(np.mean(valid_acc))+'  std= '+str(np.std(valid_acc))