import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from data import get_data

def logfunc(x, x2):
    cx = tf.clip_by_value(x, 1e-10, 1.0)
    cx2 = tf.clip_by_value(x2, 1e-10, 1.0)
    return tf.multiply(x, tf.log(tf.div(cx,cx2)))


def kl_div(p, p_hat):
    inv_p = 1 - p
    inv_p_hat = 1 - p_hat 
    return logfunc(p, p_hat) + logfunc(inv_p, inv_p_hat)

x_, y_ = get_data('degree_V2500_k50_train_label100000_2.txt', 'topk_degree_V2500_k50_train_feature100000.txt')
dim = 50

x_ = (x_ - np.amin(x_,0, keepdims=True))/ np.array((np.amax(x_,0,keepdims=True) - np.amin(x_,0,keepdims=True)), 'float32')
# x_ = (x_ - np.mean(x_,0)) / np.std(x_,0)

s = np.arange(len(x_))
np.random.shuffle(s)
x_ = x_[s]
y_ = y_[s]

# Split data into train/test = 80%/20%
train_length = int(0.8*len(x_))

x_vals_train = x_[0:train_length]
y_train = y_[0:train_length]

x_vals_test = x_[train_length:]
y_test = y_[train_length:]

# Keep only without planted clique
no_clique = np.all(y_train == [1.0, 0.0], 1)
clique = np.invert(no_clique)

x_vals_train = x_vals_train[no_clique]
x_vals_test = x_vals_test[np.all(y_test == [1.0, 0.0], 1)]

x_accuracy_test = x_[train_length:]


# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)


p = 0.01
beta = 0.03
hidden_layers = [dim, 10]
classes = 2

ops.reset_default_graph()
sess = tf.Session()

# Create Placeholders
x_data = tf.placeholder(shape=[None, dim], dtype=tf.float32)
y_target = tf.placeholder(shape= [None, classes], dtype= tf.int32)

alpha = tf.placeholder(tf.float32)


z = x_data
w = []
b = []

threshold = 0.7

reverse = False

# Encoding
for i in range(0, len(hidden_layers)-1):
    dim_in = hidden_layers[i]
    dim_out = hidden_layers[i+1]
    w.append(tf.Variable(tf.truncated_normal([dim_in,dim_out], stddev=tf.sqrt(2/float((dim_in + dim_out))))))
    z = tf.nn.sigmoid(tf.matmul(z, w[i]) + tf.Variable(tf.zeros([dim_out])))

bottleneck = z

for i in range(len(hidden_layers)-1, 0, -1):
    dim_in = hidden_layers[i]
    dim_out = hidden_layers[i-1]
    if reverse:
        w_t = tf.transpose(w[i-1])
    else:
       w_t = tf.Variable(tf.truncated_normal([dim_in,dim_out], stddev=tf.sqrt(2/float((dim_in + dim_out)))))
    z = tf.nn.sigmoid(tf.matmul(z, w_t)+ tf.Variable(tf.zeros([dim_out])))


# Loss 
p_hat = tf.reduce_mean(bottleneck, 0)
latent_loss = tf.reduce_sum(kl_div(p, p_hat))
mse_s = tf.reduce_mean(tf.square(z - x_data), 1)
loss = tf.reduce_mean(mse_s) + beta * latent_loss

targets = tf.argmax(y_target, 1)

true_false = tf.logical_and(tf.less(threshold, mse_s), tf.equal(targets, 0))
true_true = tf.logical_and(tf.less(mse_s, threshold), tf.equal(targets, 1))
acc = tf.reduce_mean(tf.cast(tf.logical_or(true_true, true_false), tf.float32))

# Optimizer
optimizer = tf.train.AdamOptimizer(alpha)
train = optimizer.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
acc_vec = []
temp_vecloss = []
temp_veclatentloss = []
batch_size = 1024
size_epoch = len(x_vals_train)//batch_size
epochs = 600
ACC_PERIOD = 100
ANOMALY_TRY = 100

for e in range(epochs):
    for i in range(size_epoch):
        step = e*size_epoch + i
        x_batch = x_vals_train[i*batch_size:(i+1)*batch_size][:, :]

        learning_rate = 0.01

        _ , temp_loss, temp_latent_loss = sess.run([train, loss, latent_loss], 
            feed_dict={alpha: learning_rate, x_data: x_batch})
        loss_vec.append(temp_loss)
        temp_vecloss.append(temp_loss)
        temp_veclatentloss.append(temp_latent_loss)

        if i % ACC_PERIOD == 0:
            avg_loss = sum(temp_vecloss) / float(len(temp_vecloss))
            avg_latent = sum(temp_veclatentloss) / float(len(temp_veclatentloss))
            test_loss = sess.run(loss, 
            feed_dict={x_data: x_vals_test})
            print('Epoch: ' + str(e)+ ' Step: ' + str(step+1) +'. Loss_training = ' + str(avg_loss))+ ' Test Loss = ' + str(test_loss)

    if e % ANOMALY_TRY == 0:
        recons = sess.run(acc, 
            feed_dict={x_data: x_accuracy_test, y_target: y_test})
        
mse_test = sess.run(mse_s, feed_dict={x_data: x_accuracy_test})

import pandas as pd
from sklearn.metrics import precision_recall_curve

error_df = pd.DataFrame({'reconstruction_error': mse_test, 'true_class': np.argmax(y_test, 1)})


precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()

plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()

threshold = 0.001

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Planted" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();
