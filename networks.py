import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import functools
import os
from itertools import product, izip
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import haikunator
from types import *
from tqdm import tqdm
from data import train_test_valid_shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

class Trainer:

	def __init__(self, args, network, trials):
		self.net = network
		self.args = args
		self.trials = trials

	def train(self, X_, y_):
		# print 'Running with ' + str(self.args)
		batch_size = self.args['batch_size']
		epochs = self.args['epochs']

		test_acc = []
		valid_acc = []

		for trial in range(self.trials):
			vals = train_test_valid_shuffle(X_, y_, train_ratio=self.args['train_ratio'], 
						import_test_data=self.args['import_test_data'], nb_samples= self.args['nb_samples'], seed=trial)
			X, y = vals[0]
			X_test, y_test = vals[1]
			X_valid, y_valid = vals[2]

			network = self.net(self.args)
			network.init()
			v , t, _ = network.train_epochs(batch_size, epochs, X, y, X_valid, y_valid, X_test, y_test)

			test_acc.append(t)
			valid_acc.append(v)

		return test_acc, valid_acc

class Optimizer:

	def my_product(self, dicts):
		return (dict(izip(dicts, x)) for x in product(*dicts.itervalues()))


	def __init__(self, search_space, network):

		self.search_space = search_space
		self.net = network

	def search(self, X, y, X_valid, y_valid, X_test, y_test):
		best_validation_accuracy = 0
		space = list(self.my_product(self.search_space))
		for current_args in tqdm(space):
			# print 'Running with ' + str(current_args)
			batch_size = current_args['batch_size']
			epochs = current_args['epochs']

			network = self.net(current_args, subdir='Grid_Search')
			network.init()
			acc_valid , acc_test, epoch = network.train_epochs(batch_size, epochs, X, y, X_valid, y_valid, X_test, y_test)

			if acc_valid > best_validation_accuracy:
				best_hyperparams = current_args
				best_validation_accuracy = acc_valid
				best_hyperparams['valid_accuracy'] = acc_valid
				best_hyperparams['test_accuracy'] = acc_test
		print best_hyperparams
		return best_hyperparams


def scope(function):
	name = function.__name__
	attribute = '_cache_' + name
	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self,attribute):
			with tf.variable_scope(name):
				setattr(self,attribute,function(self))
		return getattr(self,attribute)
	return decorator

class Network:

	def __init__(self, args, random_state=42, subdir=''):
		ops.reset_default_graph()
		np.random.seed(random_state)
		tf.set_random_seed(random_state) 

		self.batch_size = args['batch_size']
		self.classes = args['classes']
		self.run = haikunator.Haikunator().haikunate(token_length=0)
		self.name = ''
		self.data = args['data']
		self.subdir = subdir
		
		for i , (key , val) in enumerate(args.items()):
			if type(val) is TupleType or type(val) is ListType:
				if isinstance(val[0], CNN_description) or isinstance(val[0], FF_description):
					tmp_val = val
					val = ''
					for v in tmp_val:
						val += v.printit()+'_'
					val = val[:-1]
				else:
					tmp_val = val
					val = ''
					for v in tmp_val:
						val += str(v)+'_'
					val = val[:-1]
			if type(val) is FunctionType:
				val = val.__name__
			if val.__class__ == type:
				val = val.__name__
			self.name += str(key)[:3]+'_'+str(val)+'_'
		self.name = self.name[:-1]

	@scope
	def init_log(self):
		print self.subdir
		print self.run
		print self.name
		print self.data
		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(os.path.join('log', self.subdir, self.data, self.name, self.run, 'train'), tf.get_default_graph())
		self.valid_writer = tf.summary.FileWriter(os.path.join('log', self.subdir, self.data, self.name, self.run, 'valid'), tf.get_default_graph())
		self.test_writer = tf.summary.FileWriter(os.path.join('log', self.subdir, self.data, self.name, self.run, 'test'), tf.get_default_graph())

	def save_model(self, step):
		self.saver.save(self.sess, os.path.join('models', self.name, self.run), global_step=step)

	@scope
	def network(self):
		pass

	@scope
	def loss(self):
		if self.classes > 1:
			l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.y_target))
		else:
			l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.network, labels=self.y_target))
		tf.summary.scalar('value', l)
		return l

	@scope
	def optimize(self):
		my_opt = self.opt(self.learning_rate)
		return my_opt.minimize(self.loss)

	@scope
	def accuracy(self, threshold=0.5):
		if self.classes > 1:
			logits = tf.argmax(self.network, axis=1)
			targets = tf.argmax(self.y_target, axis=1)

			acc_value = tf.reduce_mean(tf.cast(tf.equal(logits, targets), tf.float32))
			acc = acc_value
		else:
			acc_value, update = tf.metrics.auc(self.y_target, tf.nn.sigmoid(self.network), num_thresholds=2000)
			acc = update

		tf.summary.scalar('value', acc_value)
		return acc

	def predict(self, X):
		val = self.sess.run(tf.nn.sigmoid(self.network), feed_dict={self.x_data: X, self.training:False})
		return val

	def train(self, X, y, n):
		summary, _  = self.sess.run([self.merged, self.optimize], feed_dict={self.x_data: X, self.y_target: y, self.training:True})
		self.train_writer.add_summary(summary, n)

	def valid(self, X, y, n):
		summary, acc =  self.sess.run([self.merged, self.accuracy], feed_dict={self.x_data: X, self.y_target: y, self.training:False})
		self.valid_writer.add_summary(summary, n)
		return acc

	def test(self, X, y, n):
		summary, acc = self.sess.run([self.merged, self.accuracy], feed_dict={self.x_data: X, self.y_target: y, self.training:False})
		self.test_writer.add_summary(summary, n)
		return acc

	def auc(self, X, y):
		if len(X.shape) != tf.rank(self.x_data).eval(session=self.sess):
			X = np.expand_dims(X, 2)
		pred = self.predict(X)
		fpr, tpr, thresholds = roc_curve(y, pred)
		a = auc(fpr, tpr)

		# plt.figure()
		# lw = 2
		# plt.plot(fpr, tpr, color='darkorange',
		#          lw=lw, label='ROC curve (area = %0.2f)' % a)
		# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.0, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title('Receiver operating characteristic example')
		# plt.legend(loc="lower right")

		# ax2 = plt.gca().twinx()
		# ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
		# ax2.set_ylabel('Threshold',color='r')
		# ax2.set_ylim([thresholds[-1],thresholds[0]])
		# ax2.set_xlim([fpr[0],fpr[-1]])

		return a

	def accuracy_binary(self, X, y):
		pred = self.predict(X)
		logits = np.array(pred >= 0.5, np.int32)
		return accuracy_score(y, logits)

	def init(self):
		with self.graph.as_default():
			init = tf.global_variables_initializer()
			self.sess.run(tf.local_variables_initializer())
		self.sess.run(init)

	def fit_epoch(self, X, y, e, e_size):
		for i in range(e_size):
			x_batch = X[i*self.batch_size:(i+1)*self.batch_size]
			y_batch = y[i*self.batch_size:(i+1)*self.batch_size]
			self.train(x_batch, y_batch, e*e_size + i)

	def fit(self, X, y, epochs):
		epoch_size = len(X)//self.batch_size
		for e in range(epochs):
			for i in range(epoch_size):
				x_batch = X[i*self.batch_size:(i+1)*self.batch_size]
				y_batch = y[i*self.batch_size:(i+1)*self.batch_size]
				self.train(x_batch, y_batch,  e*epoch_size + i)

	def train_epochs(self, batch_size, epochs, X, y, 
			X_valid, y_valid, X_test, y_test, eval_epoch=1):
		best_acc = 0
		best_epoch = 0
		best_test = 0
		epoch_size = len(X)//batch_size
		for e in range(epochs):
			self.fit_epoch(X, y, e, epoch_size)
			if e != 0 and (e+1) % eval_epoch == 0:
				n = (e+1)*epoch_size
				if self.classes == 1 :
					acc = self.auc(X_valid, y_valid)
				else:
					acc = self.valid(X_valid, y_valid, n)

				print('Epoch: ' + str(e+1) + ' accuracy = ' + str(acc))
				if acc > best_acc:
					best_acc = acc
					best_epoch = e + 1
					if self.classes == 1 :
						# acc_test = self.auc(X_test, y_test)
						acc_test = self.auc(X_test, y_test)
						print acc_test
					else:
						acc_test = self.test(X_test, y_test, n)
					best_test = acc_test

		return best_acc, best_test, best_epoch



class RNN(Network):

	def __init__(self, args, subdir=''):

		Network.__init__(self, args, subdir=subdir)
		self.graph = tf.Graph()

		self.hidden = args['hidden']
		self.layers = args['layers']
		self.dropout = args['dropout']
		self.learning_rate = args['learning_rate']
		self.opt = args['optimizer']
		self.dim = args['input_dim']
		self.classes = args['classes']
		self.blstm = args['blstm']

		self.name = 'RNN_' + self.name

		with self.graph.as_default():
			# Placeholders
			self.x_data = tf.placeholder(shape=[None, None, self.dim], dtype=tf.float32)
			self.y_target = tf.placeholder(shape=[None, self.classes], dtype=tf.float32)
			self.training = tf.placeholder(tf.bool)

			self.network
			self.loss
			self.optimize
			self.accuracy
			self.init_log

		self.sess = tf.Session(graph=self.graph)

	@scope
	def network(self):
			if not self.blstm:
				cells = []
				for _ in range(self.layers):
					cell = tf.contrib.rnn.BasicLSTMCell(self.hidden) # Or LSTMCell(num_units)
					cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
					cells.append(cell)
				network = tf.contrib.rnn.MultiRNNCell(cells)
				z, _ = tf.nn.dynamic_rnn(network, self.x_data, dtype=tf.float32)
			else:
				z = self.x_data
				for i in range(self.layers):
				    z = self.blstm_net(z, self.hidden, i, self.dropout)

			last = z[:, -1, :]
			prediction = tf.layers.dense(last, self.classes)

			return prediction

	def blstm_net(self, input_tensor, hid, i, kp):
		forward_input = input_tensor
		backward_input = tf.reverse(input_tensor, [1])

		with tf.variable_scope('forward_'+str(i)):
			# Forward pass
			forward_lstm = tf.contrib.rnn.BasicLSTMCell(hid//2)
			forward_lstm = tf.contrib.rnn.DropoutWrapper(forward_lstm, kp, kp, kp)
			forward_out, _ = tf.nn.dynamic_rnn(forward_lstm, forward_input, dtype=tf.float32)

		with tf.variable_scope('backward_'+str(i)):
			# backward pass
			backward_lstm = tf.contrib.rnn.BasicLSTMCell(hid//2)
			backward_lstm = tf.contrib.rnn.DropoutWrapper(backward_lstm, kp, kp, kp)
			backward_out, _ = tf.nn.dynamic_rnn(backward_lstm, backward_input, dtype=tf.float32)

		# Concatenate the RNN outputs and return
		return tf.concat([forward_out[:,:,:], backward_out[:,::-1,:]], 2)


class DNN(Network):

	def __init__(self, args, subdir=''):

		Network.__init__(self, args, subdir=subdir)
		self.graph = tf.Graph()

		self.classes = args['classes']
		self.hidden = args['hidden']
		self.dropout = args['dropout']
		self.learning_rate = args['learning_rate']
		self.opt = args['optimizer']
		self.input_dim = args['input_dim']
		self.activation = args['activation']

		self.name = 'DNN_' + self.name

		with self.graph.as_default():
			self.x_data = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
			self.y_target = tf.placeholder(shape=[None, self.classes], dtype=tf.float32)
			self.training = tf.placeholder(tf.bool)

			self.network
			self.loss
			self.optimize
			self.accuracy

			self.init_log

		self.sess = tf.Session(graph=self.graph)

	@scope
	def network(self):
		# Create Placeholders
		z = self.x_data
		for layer in self.hidden:
			z = tf.layers.dense(z, layer, activation = self.activation)
			z = tf.layers.dropout(z, rate= self.dropout, training = self.training)

		return tf.layers.dense(z, self.classes)


class CNN_description:
	def __init__(self, filters, size, activation):
		self.filters = filters
		self.size = size
		self.activation = activation

	def printit(self):
		return str(self.filters)+'_'+str(self.size[0])+'x'+str(self.size[1])+'_'+str(self.activation.__name__ if self.activation is not None else 'None')

class FF_description:
	def __init__(self, hidden, activation):
		self.hidden = hidden
		self.activation = activation
	def printit(self):
		return str(self.hidden)+'_'+str(self.activation.__name__ if self.activation is not None else 'None')



class CNN(Network):

	def __init__(self, args):

		Network.__init__(self, args)
		self.graph = tf.Graph()

		self.height, self.width = args['shape']
		self.ffcs = args['ffcs']
		self.batch_norm = args['batch_norm']
		self.layers = args['layers']
		self.dropout = args['dropout']
		self.batch_size = args['batch_size']
		self.learning_rate = args['learning_rate']
		self.opt = args['optimizer']
		self.classes = args['classes']

		self.name = 'CNN_' + self.name


		with self.graph.as_default():
			self.x_data = tf.placeholder(shape=[None, self.height, self.width, 1], dtype=tf.float32)
			self.y_target = tf.placeholder(shape=[None, self.classes], dtype=tf.float32)
			self.training = tf.placeholder(tf.bool)

			self.network
			self.loss
			self.optimize
			self.accuracy
			self.init_log

		self.sess = tf.Session(graph=self.graph)

	def conv_layer(self, input_tensor, filters, size, activation):

		z = tf.layers.conv2d(
				inputs=input_tensor,
				filters=filters,
				kernel_size=size,
				padding="same",
				activation= None if self.batch_norm else activation)

		if self.batch_norm:
			z = tf.contrib.layers.batch_norm(
					z, 
					center=True, 
					scale=True, 
					is_training=self.training)
			z = activation(z)

		return z

	def maxpool(self, input_tensor, size):
		return tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[size, size], strides=size)

	@scope
	def network(self):
		z = self.x_data

		for i, archi in enumerate(self.layers):
			if i > 0: z = self.maxpool(z , 2)
			z = self.conv_layer(z, archi.filters, archi.size, archi.activation)

		z = tf.layers.flatten(z)
		# z = tf.reduce_mean(z, [1,2])
		print 'FLatten ', z

		for i, ffc in enumerate(self.ffcs):
			if i > 0: z = tf.layers.dropout(inputs=z, rate=self.dropout, training=self.training)
			z = tf.layers.dense(z, ffc.hidden, activation=ffc.activation)

		return z


		my_opt = self.opt(self.learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_step = my_opt.minimize(self.loss)

		logits = tf.argmax(prediction, axis=1)
		targets = tf.argmax(self.y_target, axis=1)

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, targets), tf.float32))