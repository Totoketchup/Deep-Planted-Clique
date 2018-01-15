import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from data import get_data, get_numpy_data, get_h5_data
from sklearn.preprocessing import StandardScaler

# x_vals, y_vals = get_data('degree_V2500_k50_train_label100000_2.txt', 'topk_degree_V2500_k50_train_feature100000.txt')
# dim = 50
# feature_dim = 1
# seq_size = dim

# x_vals, y_vals = get_numpy_data('clique-N1000-K31-E50-labels.npy','clique-N1000-K31-E50-features.npy')
x_vals, y_vals = get_h5_data(N=1000, K=25, E=10, M=1, ex=True, L=False, fl=False, one_hot=True)
shape = x_vals.shape

x_vals = np.reshape(x_vals, (shape[0], -1, shape[-1]))

x_vals = x_vals[:, :, :]

# x_vals = np.transpose(x_vals, (0, 2, 1))
x_vals = (x_vals - np.mean(x_vals,0)) / np.std(x_vals,0)
N, height, width = x_vals.shape
print x_vals.shape
# x_vals = (x_vals - np.amin(x_vals,0, keepdims=True))/ np.array((np.amax(x_vals,0,keepdims=True) - np.amin(x_vals,0,keepdims=True)), 'float32')

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

hidden = 50
layers = 4
classes = 2


ops.reset_default_graph()
sess = tf.Session()

# Create Placeholders
x_data = tf.placeholder(shape=[None, height, width, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, classes], dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32)
alpha = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

conv1 = tf.layers.conv2d(
    inputs=x_data,
    filters=4,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.sigmoid)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=8,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.sigmoid)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=16,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.sigmoid)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

pool3_flat = tf.contrib.layers.flatten(pool3)
print pool3_flat
# Softmax layer.
dense1 = tf.layers.dense(pool3_flat, 32, activation=tf.nn.sigmoid)
dropout = tf.layers.dropout(inputs=dense1, rate=keep_prob, training=training)
dense2 = tf.layers.dense(dropout, 16, activation=tf.nn.sigmoid)
dropout2 = tf.layers.dropout(inputs=dense2, rate=keep_prob, training=training)
prediction = tf.layers.dense(dropout2, classes)


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

batch_size = 512
size_epoch = len(x_vals_train)//batch_size
epochs = 1000
ACC_PERIOD = 10

for e in range(epochs):
    for i in range(size_epoch):
        step = e*size_epoch + i
        x_batch = x_vals_train[i*batch_size:(i+1)*batch_size][:,:,:,np.newaxis]
        y_batch = y_vals_train[i*batch_size:(i+1)*batch_size]

        learning_rate = 0.001 

        _ , temp_loss, u = sess.run([train_step, loss, accuracy], 
            feed_dict={alpha: learning_rate, x_data: x_batch, y_target: y_batch, keep_prob: 0.9, training:True})
        loss_vec.append(temp_loss)
        temp_vecloss.append(temp_loss)
        temp_vecacc.append(u)
        # test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test)})
        # test_loss.append(test_temp_loss)
        if (step) % ACC_PERIOD == 0:
            
            acc = sess.run(accuracy, 
                feed_dict={x_data: x_vals_test[:,:,:,np.newaxis], 
                y_target: y_vals_test, 
                keep_prob: 1.0,
                training:False})
            acc_vec.append(acc)
            avg = sum(temp_vecloss) / float(len(temp_vecloss))
            avg_acc = sum(temp_vecacc) / float(len(temp_vecacc))
            print('Epoch: ' + str(e)+ ' Step: ' + str(step+1) + '. Loss = ' + str(avg) + ' accuracy = ' + str(acc)+ ' train_acc = '+ str(u))
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

