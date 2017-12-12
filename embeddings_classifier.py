
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
import json
import os

#
# CONFIG
E = 512
N = 1000
todel = '../data/kdd_datasets/clique/'

x_vals = np.empty((N,E), 'float32')
y_vals = np.empty((N,), 'int32')

##
## OPEN DATA / EMBEDDINGS
data_path = os.path.join('graph2vec_tf-master','embeddings','clique_dims_512_epochs_1000_lr_0.1_embeddings.txt')
with open(data_path) as json_data:
    data = json.load(json_data)

for d in data:
    x_vals[int(d[len(todel):-10])] = data[d]

labels_path = os.path.join('graph2vec_tf-master','data','kdd_datasets','clique.Labels')
with open(labels_path) as labels_data:
    lines = labels_data.readlines()
    for i, line in enumerate(lines):
        y_vals[i] = int(line.split(' ')[1][:-1])

y_vals = np.array([[1.,0.] if y == 1. else [0., 1.] for y in y_vals])
print y_vals

dim = E

# x_vals = (x_vals - np.min(x_vals,0)) / (np.array(np.max(x_vals,0) - np.min(x_vals,0), 'float32'))


# reset the graph for new run
ops.reset_default_graph()

# Create graph session 
sess = tf.Session()

# set batch size for training
batch_size = 8

# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

s = np.arange(len(x_vals))
np.random.shuffle(s)

x_vals = x_vals[s]
y_vals = y_vals[s]

# Split data into train/test = 80%/20%
train_length = int(0.8*len(x_vals))

x_vals_train = x_vals[0:train_length]
y_vals_train = y_vals[0:train_length]

x_vals_test = x_vals[train_length:]
y_vals_test = y_vals[train_length:]


# Create Placeholders
x_data = tf.placeholder(shape=[None, dim], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)
training = tf.placeholder(tf.bool)
alpha = tf.placeholder(tf.float32)



droprate = 0.0
# regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)

z = tf.layers.dense(x_data, 512, activation = tf.nn.sigmoid)
z = tf.layers.dropout(z, rate= 0.5, training=training)
z = tf.layers.dense(z, 128, activation = tf.nn.sigmoid)
z = tf.layers.dropout(z, rate= 0.5, training=training)
z = tf.layers.dense(z, 64, activation = tf.nn.sigmoid)
z = tf.layers.dropout(z, rate= 0.5, training=training)
final_output = tf.layers.dense(z, 2)

# Declare loss function (L1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y_target))
# Declare optimizer
my_opt = tf.train.AdamOptimizer(alpha)
train_step = my_opt.minimize(loss)

logits = tf.argmax(final_output, axis=1)
targets = tf.argmax(y_target, axis=1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, targets), tf.float32))

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
acc_vec = []
print(y_vals)

size_epoch = len(x_vals_train)//batch_size
epochs = 600
ACC_PERIOD = 25

for e in range(epochs):
    for i in range(size_epoch):
        step = e*size_epoch + i
        x_batch = x_vals_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_vals_train[i*batch_size:(i+1)*batch_size]

        learning_rate = 0.001
        # if step > 4000:
        #     learning_rate = 0.01 

        _ , temp_loss, u = sess.run([train_step, loss, accuracy], feed_dict={alpha: learning_rate, x_data: x_batch, y_target: y_batch, training: True})
        loss_vec.append(temp_loss)
        # test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test)})
        # test_loss.append(test_temp_loss)
        if (step) % ACC_PERIOD == 0:
            
            acc = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test, training: False})
            acc_vec.append(acc)
            print('Step: ' + str(step+1) + '. Loss = ' + str(temp_loss) + ' accuracy = ' + str(acc))


# Plot loss (MSE) over time
t_loss = np.arange(0, len(loss_vec))
t_acc = np.arange(0, len(loss_vec), ACC_PERIOD)
print t_acc.shape, len(acc_vec)
plt.plot(t_loss, loss_vec, 'k-', label='Train Loss')
plt.plot(t_acc, acc_vec, 'r--', label='Test Loss')
plt.axis([0, len(loss_vec), 0, 1.0])
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()