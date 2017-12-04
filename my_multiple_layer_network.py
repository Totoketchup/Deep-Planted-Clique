# Using a Multiple Layer Network
#---------------------------------------
#
# We will illustrate how to use a Multiple
# Layer Network in TensorFlow
#
# Low Birthrate data:
#
#Columns    Variable                                              Abbreviation
#-----------------------------------------------------------------------------
# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
#                          1 = Birth Weight < 2500g)
# Age of the Mother in Years                              AGE
# Weight in Pounds at the Last Menstrual Period           LWT
# Race (1 = White, 2 = Black, 3 = Other)                  RACE
# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
# History of Premature Labor (0 = None  1 = One, etc.)    PTL
# History of Hypertension (1 = Yes, 0 = No)               HT
# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
# Birth Weight in Grams                                   BWT
#------------------------------
# The multiple neural network layer we will create will be composed of
# three fully connected hidden layers, with node sizes 50, 25, and 5

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import random
import math
import requests
from tensorflow.python.framework import ops
from sklearn.preprocessing import OneHotEncoder

f = open('degree_V2500_k50_train_label100000_2.txt')
#f = open('topk_degree_V2500_k39_train_label5000_3.txt')
lines2 = f.readlines()
f.close()

train_labels = lines2[0].split(' ')

for i in range(0, len(train_labels)-1):
    train_labels[i] = int(train_labels[i])
#print(len(train_labels))
del train_labels[len(train_labels)-1]

y_vals = np.array(train_labels)
y_vals = np.array([[1.,0.] if y == 1. else [0., 1.] for y in y_vals])

# Filter for features of interest
#cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
#x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

dim = 50
f = open('topk_degree_V2500_k50_train_feature100000.txt')
#f = open('topk_degree_V2500_k50_train_feature5000_3.txt')
lines2 = f.readlines()
f.close()

data = np.arange(100000*dim)
num = 0
for line in lines2:
    my_list = line.split(',')
    for i in range(0, len(my_list)):
        data[num] = (int(my_list[i]))
        num = num+1
    #del my_list[len(my_list)-1]

x_vals = np.reshape(data,(100000,dim))

print x_vals
x_vals = (x_vals - np.mean(x_vals,0)) / np.var(x_vals,0)

print x_vals

# reset the graph for new run
ops.reset_default_graph()

# Create graph session 
sess = tf.Session()

# set batch size for training
batch_size = 2048

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

z = tf.layers.dense(x_data, 20, activation = tf.nn.relu)
z = tf.layers.dropout(z, rate= 0.0, training=training)
z = tf.layers.dense(z, 20, activation = tf.nn.relu)
z = tf.layers.dropout(z, rate= 0.0, training=training)
# z = tf.layers.dense(z, 100, activation = tf.nn.relu)
# z = tf.layers.dropout(z, rate= 0.2, training=training)
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