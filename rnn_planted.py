import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from data import get_data

def blstm(input_tensor, hid, i):
    forward_input = input_tensor
    backward_input = tf.reverse(input_tensor, [1])

    with tf.variable_scope('forward_'+str(i)):
        # Forward pass
        forward_lstm = tf.contrib.rnn.BasicLSTMCell(hid//2)
        forward_lstm = tf.contrib.rnn.DropoutWrapper(forward_lstm, keep_prob, keep_prob, keep_prob)
        forward_out, _ = tf.nn.dynamic_rnn(forward_lstm, forward_input, dtype=tf.float32)

    with tf.variable_scope('backward_'+str(i)):
        # backward pass
        backward_lstm = tf.contrib.rnn.BasicLSTMCell(hid//2)
        backward_lstm = tf.contrib.rnn.DropoutWrapper(backward_lstm, keep_prob, keep_prob, keep_prob)
        backward_out, _ = tf.nn.dynamic_rnn(backward_lstm, backward_input, dtype=tf.float32)

    # Concatenate the RNN outputs and return
    return tf.concat([forward_out[:,:,:], backward_out[:,::-1,:]], 2)

x_vals, y_vals = get_data('degree_V2500_k50_train_label100000_2.txt', 'topk_degree_V2500_k50_train_feature100000.txt')
dim = 50
# x_vals = (x_vals - np.amin(x_vals,0, keepdims=True))/ np.array((np.amax(x_vals,0,keepdims=True) - np.amin(x_vals,0,keepdims=True)), 'float32')
x_vals = (x_vals - np.mean(x_vals,0)) / np.std(x_vals,0)



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

hidden = 10
layers = 2
classes = 2
feature_dim = 1
seq_size = dim

ops.reset_default_graph()
sess = tf.Session()

# Create Placeholders
x_data = tf.placeholder(shape=[None, seq_size, feature_dim], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, classes], dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32)
alpha = tf.placeholder(tf.float32)


# cells = []
# for _ in range(layers):
#     cell = tf.contrib.rnn.BasicLSTMCell(hidden) # Or LSTMCell(num_units)
#     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
#     cells.append(cell)
# network = tf.contrib.rnn.MultiRNNCell(cells)
# output, _ = tf.nn.dynamic_rnn(network, x_data, dtype=tf.float32)


l = blstm(x_data, 20,1)
output = blstm(l, 20,2)

# Select last output.
last = output[:, -1, :]
# Softmax layer.
prediction = tf.layers.dense(last, classes)

# Declare loss function (L1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_target))


my_opt = tf.train.AdamOptimizer(alpha)
train_step = my_opt.minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_target, 1), tf.argmax(prediction, 1)), tf.float32))
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
acc_vec = []
temp_vecloss = []
temp_vecacc = []

batch_size = 1024
size_epoch = len(x_vals_train)//batch_size
epochs = 600
ACC_PERIOD = 10

for e in range(epochs):
    for i in range(size_epoch):
        step = e*size_epoch + i
        x_batch = x_vals_train[i*batch_size:(i+1)*batch_size][:, :, np.newaxis]
        y_batch = y_vals_train[i*batch_size:(i+1)*batch_size]

        learning_rate = 0.001
        # if step > 4000:
        #     learning_rate = 0.01 

        _ , temp_loss, u = sess.run([train_step, loss, accuracy], 
            feed_dict={alpha: learning_rate, x_data: x_batch, y_target: y_batch, keep_prob: 0.7})
        loss_vec.append(temp_loss)
        temp_vecloss.append(temp_loss)
        temp_vecacc.append(u)
        # test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test)})
        # test_loss.append(test_temp_loss)
        if (step) % ACC_PERIOD == 0:
            
            acc = sess.run(accuracy, 
                feed_dict={x_data: x_vals_test[:1000, :, np.newaxis], 
                y_target: y_vals_test[:1000, :], 
                keep_prob: 1.0})
            acc_vec.append(acc)
            avg = sum(temp_vecloss) / float(len(temp_vecloss))
            avg_acc = sum(temp_vecacc) / float(len(temp_vecacc))
            print('Step: ' + str(step+1) + '. Loss = ' + str(avg) + ' accuracy = ' + str(acc)+ ' train_acc = '+ str(u))
            temp_vecloss = []

# Plot loss (MSE) over time
t_loss = np.arange(0, len(loss_vec))
t_acc = np.arange(0, len(loss_vec), ACC_PERIOD)
plt.plot(t_loss, loss_vec, 'k-', label='Train Loss')
plt.plot(t_acc, acc_vec, 'r--', label='Test Loss')
plt.axis([0, len(loss_vec), 0, 1.0])
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()