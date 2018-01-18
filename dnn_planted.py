import tensorflow as tf
import numpy as np
from data import get_data, train_test_valid_shuffle,get_txt_data
import argparse
import os
from itertools import product
# make results reproducible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from networks import DNN, Optimizer, Trainer


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
			'classes' : [2],
			'input_dim' : [input_dim],
			'activation' : [tf.nn.sigmoid],
			'data' : [args.data]
		}

		search_grid = Optimizer(search_space, DNN)
		search_grid.search(x_vals_train, y_vals_train, 
						x_vals_valid, y_vals_valid, 
						x_vals_test, y_vals_test)

	else:
		print 'Multiple trials runs'

		#clique-N1000-K30-E0-M1-exTrue-L_False-F_False/
		#DNN_optimizer_AdamOptimizer_learning_rate_0.01_batch_size_512_
		#epochs_400_classes_2_input_dim_30_dropout_0.3_hidden_200_200_data_
		#clique-N1000-K30-E0-M1-exTrue-L_False-F_False_activation_sigmoid

		#python dnn_planted.py --data local_topk_degree_V1000_k30_train_50000 --text --trials 10
		#Multiple trials runs
		#On 10 Trials:
		#Test mean = 0.631532  std= 0.0145965
		#Valid mean = 0.648995  std= 0.0132133
		
		params = {
			'hidden' : [200, 200],
			'dropout' : 0.3,
			'learning_rate' : 0.01,
			'batch_size' : 512,
			'optimizer' : tf.train.AdamOptimizer,
			'epochs' : 400,
			'classes' : 2,
			'input_dim' : input_dim,
			'data' : args.data,
			'activation' : tf.nn.sigmoid,
		}

		trials = args.trials


		test_acc, valid_acc = Trainer(params, DNN, trials).train(x_vals, y_vals)

		print 'On '+str(trials)+' Trials:'
		print 'Test mean = '+str(np.mean(test_acc))+'  std= '+str(np.std(test_acc))
		print 'Valid mean = '+str(np.mean(valid_acc))+'  std= '+str(np.std(valid_acc))
