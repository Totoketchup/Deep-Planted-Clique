import numpy as np

def get_data(label_fn, features_fn, dim=50, nb=100000):
	f = open(label_fn)
	lines2 = f.readlines()
	f.close()

	train_labels = lines2[0].split(' ')

	for i in range(0, len(train_labels)-1):
	    train_labels[i] = int(train_labels[i])
	del train_labels[len(train_labels)-1]

	y_vals = np.array(train_labels)
	y_vals = np.array([[1.,0.] if y == 1. else [0., 1.] for y in y_vals])

	f = open(features_fn)
	lines = f.readlines()
	f.close()

	x_vals = np.zeros((nb, dim))
	for i, line in enumerate(lines):
	    features = map(int, line.split(','))
	    x_vals[i] = np.array(features)

	return x_vals, y_vals
