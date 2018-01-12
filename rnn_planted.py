import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from data import get_data, get_numpy_data, get_h5_data
from itertools import product
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def blstm(input_tensor, hid, i, kp):
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

####
#### GET THE DATA 
####

# x_vals, y_vals = get_h5_data(100,10,0,2,True,fl=True)
# x_vals, y_vals = get_topological_data(100,10)
x_vals, y_vals = get_h5_data(N=100, K=10, E=0, M=1, ex=True, L=False, fl=False, one_hot=True)
print x_vals.shape


x_vals = (x_vals - np.mean(x_vals,0)) / np.std(x_vals,0)
# x_vals = x_vals[:, :-2]
_ , _ , dim  = x_vals.shape
_, classes = y_vals.shape

####
#### DATA SPLITTING / SHUFFLING
####

s = np.arange(len(x_vals))
np.random.shuffle(s)

x_vals = x_vals[s]
y_vals = y_vals[s]

# Split data into train/test/validation = 80%/10%/10%
train_length = int(0.8*len(x_vals))

x_vals_train = x_vals[:train_length]
y_vals_train = y_vals[:train_length]

x_vals_test_valid = x_vals[train_length:]
y_vals_test_valid = y_vals[train_length:]

test_length = int(0.5*len(x_vals_test_valid))

x_vals_test = x_vals_test_valid[:test_length]
y_vals_test = y_vals_test_valid[:test_length]

x_vals_valid = x_vals_test_valid[test_length:]
y_vals_valid = y_vals_test_valid[test_length:]


####
#### HYPERPARAMETERS SPACE SEARCH
####

hidden = [10, 20, 50]
layers = [1, 2, 3]
dropout = [0.8, 0.5, 0.2]
learning_rate = [0.01, 0.001]
batch_size = [1024, 2048]
optimizer = [tf.train.AdamOptimizer]

# ###
### one shot
###

# h1 = [20]
# h2 = [10,5]
# dropout = [0.2]
# learning_rate = [0.001]
# batch_size = [1024]
# optimizer = [tf.train.AdamOptimizer]


epochs = 100
eval_epoch = 10

####
#### NETWORK CLASS
####

class RNN:

    def __init__(self, hyperparams, random_state=42):

        self.graph = tf.Graph()

        ops.reset_default_graph()
        np.random.seed(random_state)
        tf.set_random_seed(random_state)
        self.hidden = hyperparams['hidden']
        self.layers = hyperparams['layers']
        self.dropout = hyperparams['dropout']
        self.batch_size = hyperparams['batch_size']
        self.learning_rate = hyperparams['learning_rate']
        self.opt = hyperparams['optimizer']

        self.network()

        self.sess = tf.Session(graph=self.graph)

    def network(self):
        with self.graph.as_default():
            # Create Placeholders
            self.x_data = tf.placeholder(shape=[None, None, dim], dtype=tf.float32)
            self.y_target = tf.placeholder(shape=[None, classes], dtype=tf.float32)
            self.training = tf.placeholder(tf.bool)


            # cells = []
            # for _ in range(self.layers):
            #     cell = tf.contrib.rnn.BasicLSTMCell(self.hidden) # Or LSTMCell(num_units)
            #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
            #     cells.append(cell)
            # network = tf.contrib.rnn.MultiRNNCell(cells)
            # output, _ = tf.nn.dynamic_rnn(network, self.x_data, dtype=tf.float32)

            z = self.x_data
            for i in range(self.layers):
                z = blstm(self.x_data, self.hidden, 1, self.dropout)
            # Select last output.
            last = z[:, -1, :]
            prediction = tf.layers.dense(last, classes)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y_target))

            my_opt = self.opt(self.learning_rate)
            self.train_step = my_opt.minimize(self.loss)

            logits = tf.argmax(prediction, axis=1)
            targets = tf.argmax(self.y_target, axis=1)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, targets), tf.float32))

        
    def train(self, X, y):
        self.sess.run(self.train_step, feed_dict={self.x_data: X, self.y_target: y, self.training:True})

    def test(self, X, y):
        return self.sess.run(self.accuracy, feed_dict={self.x_data: X, self.y_target: y, self.training:False})

    def init(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit_epoch(self, X, y, e, e_size):
        for i in range(e_size):
            x_batch = X[i*self.batch_size:(i+1)*self.batch_size]
            y_batch = y[i*self.batch_size:(i+1)*self.batch_size]
            network.train(x_batch, y_batch)

    def fit(self, X, y, epochs):
        epoch_size = len(X)//self.batch_size
        for e in range(epochs):
            for i in range(epoch_size):
                x_batch = X[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y[i*self.batch_size:(i+1)*self.batch_size]
                network.train(x_batch, y_batch)

####
#### GRID SEARCH
####

best_validation_accuracy = 0.0
best_hyperparams = {'h1':0,'h2':0,'dropout':0,'learning_rate':0,'batch_size':0, 'optimizer':0}

for _hidden, _layers, _dropout, _learning_rate, _batch_size, _optimizer in product(hidden, layers, dropout, learning_rate, batch_size, optimizer):
    hyperparams = { 'hidden':_hidden,
                    'layers':_layers,
                    'dropout':_dropout,
                    'learning_rate':_learning_rate,
                    'batch_size':_batch_size, 
                    'optimizer':_optimizer,
                    'epochs':0}
    epoch_size = len(x_vals_train)//_batch_size


    print 'Running with ' + str(hyperparams)

    network = RNN(hyperparams)
    network.init()

    for e in range(epochs):
        network.fit_epoch(x_vals_train, y_vals_train, e, epoch_size)
        if e != 0 and (e+1) %eval_epoch == 0:
            acc = network.test(x_vals_valid, y_vals_valid)
            print('Epoch: ' + str(e+1) + ' accuracy = ' + str(acc))
            if acc > best_validation_accuracy:
                print 'BEST'
                best_validation_accuracy = acc
                best_hyperparams['epochs'] = e+1
                best_hyperparams['valid_accuracy'] = acc


print 'The best hyperparameters set is:'
print str(best_hyperparams)
network = RNN(best_hyperparams)
network.init()
network.fit(x_vals_train, y_vals_train, best_hyperparams['epochs'])
acc = network.test(x_vals_test, y_vals_test)
print ' Test Accuracy = ' + str(acc)



# {'optimizer': <class 'tensorflow.python.training.adam.AdamOptimizer'>, 'h2': 50, 
# 'dropout': 0.6, 'h1': 20, 'batch_size': 2048, 'epochs': 300, 'learning_rate': 0.01}
# Test Accuracy = 0.6655
# Valid Accuracy : 0.671
