import numpy as np
import h5py
import os
import pandas as pd

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

#local_topk_degree_V1000_k30_train_50000feature
def get_txt_data(data, data_path, one_hot=True):
	label_fn = os.path.join(data_path, data+'label.txt')
	features_fn = os.path.join(data_path, data+'feature.txt')


	k_info = data.split("_k")[1]
	i = 1
	while RepresentsInt(k_info[:i]):
		i+=1
	dim = int(k_info[:i-1])


	while RepresentsInt(data[-i:]):
		i+=1
	nb = int(data[-i+1:])


	f = open(label_fn)
	lines2 = f.readlines()
	f.close()

	y_vals = map(int, lines2)

	if one_hot:
		y_vals = np.array([[1.,0.] if y == 1. else [0., 1.] for y in y_vals])
	else:
		y_vals = np.array(y_vals)

	f = open(features_fn)
	lines = f.readlines()
	f.close()

	x_vals = np.zeros((nb, dim))
	for i, line in enumerate(lines):
	    features = map(int, line.split(','))
	    x_vals[i] = np.array(features)[::-1]

	x_vals = (x_vals - np.mean(x_vals,0)) / np.std(x_vals,0)
	return x_vals, y_vals


def get_numpy_data(labels_fn, features_fn, one_hot=True):
	labels = np.load(labels_fn)
	features = np.load(features_fn)

	if one_hot:
		s = pd.Series(labels)
		y = np.array(pd.get_dummies(s), 'float32')
	else:
		y = labels

	return features, y

def get_h5_data(N, K, E, M, ex, L=True, fl=False, one_hot=True):

	name = "clique-N{}-K{}-E{}-M{}-ex{}-L:{}-F:{}".format(N, K, E, M, ex, L, fl)
	print name
	h5 = h5py.File('data/'+name+'.h5')

	features = h5['features']
	labels = h5['labels']
	if one_hot:
		s = pd.Series(labels)
		y = np.array(pd.get_dummies(s), 'float32')
	else:
		y = labels
	return features, y

def get_h5_by_name(path, name, one_hot=True):

	file_path = os.path.join(path,name+'.h5',)
	h5 = h5py.File(file_path)

	features = h5['features']
	labels = h5['labels']
	if one_hot:
		s = pd.Series(labels)
		y = np.array(pd.get_dummies(s), 'float32')
	else:
		y = labels
	return features, y

def get_topological_data(N, K):

	name = "clique-N{}-K{}".format(N, K)
	h5 = h5py.File('data/'+name+'.h5')

	features = h5['features']
	labels = h5['labels']
	s = pd.Series(labels)
	y = np.array(pd.get_dummies(s), 'float32')

	return features, y

def get_data_by_name(path, name, one_hot=True):

	file_path = os.path.join(path,name+'.h5',)
	h5 = h5py.File(file_path)

	features = h5['features']
	labels = h5['labels']

	if one_hot:
		s = pd.Series(labels)
		y = np.array(pd.get_dummies(s), 'float32')
	else:
		y = np.expand_dims(np.array(labels),1)
		print y
	return features, y

def get_data(data, data_path, topological, feature_truc=0, one_hot=True):
	if not topological:
		x_vals, y_vals = get_data_by_name(data_path, data, one_hot)
	else:
		x_vals, y_vals = get_data_by_name(data_path, data, False)

	# x_vals = x_vals[:,:47,:47]

	x_vals = (x_vals - np.mean(x_vals,0)) / np.std(x_vals,0)

	return x_vals, y_vals

def get_data_adjacency(data, data_path, one_hot=True):

	file_path = os.path.join(data_path,data+'.h5',)
	h5 = h5py.File(file_path)

	features = h5['features']
	labels = h5['labels']

	if one_hot:
		s = pd.Series(labels)
		y = np.array(pd.get_dummies(s), 'float32')
	else:
		y = np.expand_dims(np.array(labels),1)
	
	return np.array(features), y



def train_test_valid_shuffle(x_vals, y_vals, train_ratio=0.8, valid_ratio=0.1, 
		test_ratio=0.1, nb_samples=0, import_test_data='', seed=42):
	
	# Shuffle
	s = np.arange(len(x_vals))
	np.random.seed(seed*42)
	np.random.shuffle(s)

	x_vals = x_vals[s]
	y_vals = y_vals[s]

	# Truncate if necessary
	if nb_samples != 0:
		x_vals = x_vals[:nb_samples]
		y_vals = y_vals[:nb_samples]

	# Split data into train/test/validation = 80%/10%/10%
	train_length = int(train_ratio*len(x_vals))

	x_vals_train = x_vals[:train_length]
	y_vals_train = y_vals[:train_length]

	x_vals_test_valid = x_vals[train_length:]
	y_vals_test_valid = y_vals[train_length:]

	if import_test_data == '':
		test_length = int((test_ratio/(1-train_ratio))*len(x_vals_test_valid))

		# print test_ratio/(1-train_ratio) + valid_ratio/(1-train_ratio)
		assert abs(test_ratio/(1-train_ratio) + valid_ratio/(1-train_ratio) - 1.0) < 0.001

		x_vals_test = x_vals_test_valid[:test_length]
		y_vals_test = y_vals_test_valid[:test_length]

		x_vals_valid = x_vals_test_valid[test_length:]
		y_vals_valid = y_vals_test_valid[test_length:]
	else:
		x_vals_valid = x_vals_test_valid
		y_vals_valid = y_vals_test_valid

		x_vals_test, y_vals_test = get_txt_data(import_test_data, 'data', False)

	print x_vals_train.shape, x_vals_valid.shape, x_vals_test.shape
	return (x_vals_train, y_vals_train), (x_vals_test, y_vals_test), (x_vals_valid, y_vals_valid)

#TEST
if __name__ == "__main__":
   print get_h5_data(100,10,10,2,2) 