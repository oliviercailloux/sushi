import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


NB_USERS = 5000
FEAT_USER = 1
NB_ITEMS = 100
FEAT_ITEM = 4

NB_EPOCHS = 500
LAMBDA_REG = 1e-5
# LAMBDA_REG = 0
# learning_rate = 0.001

# Load data
users = pd.read_csv('data/sushi/sushi3.udata', sep='\t', names=('uid', 'gender', 'age', 'time', 'old_prefecture', 'old_region', 'old_eastwest', 'prefecture', 'region', 'eastwest', 'same'))
items = pd.read_csv('data/sushi/sushi3.idata', sep='\t', names=('iid', 'name', 'style', 'major', 'minor', 'heaviness', 'frequency', 'price', 'popularity'))
R = pd.read_csv('data/sushi/sushi3b.5000.10.score', sep=' ', header=None)
triplets = []
for i, line in enumerate(np.array(R)):
    for j, v in enumerate(line):
        if v != -1:
            triplets.append((i, j, v))
df_ratings = pd.DataFrame(triplets, columns=('user', 'item', 'rating'))
train, test = train_test_split(df_ratings, test_size=0.2, shuffle=True)

# TF
A = tf.constant(np.array(users[['age']]).astype(np.float32))
B = tf.constant(np.array(items[['heaviness', 'frequency', 'price', 'popularity']]).astype(np.float32))

W_V = tf.get_variable('W_V', shape=[NB_ITEMS, FEAT_USER], dtype=np.float32, initializer=tf.truncated_normal_initializer(stddev=1))
W_U = tf.get_variable('W_U', shape=[NB_USERS, FEAT_ITEM], dtype=np.float32, initializer=tf.truncated_normal_initializer(stddev=1))
M = tf.get_variable('M', shape=[FEAT_USER, FEAT_ITEM], dtype=np.float32, initializer=tf.truncated_normal_initializer(stddev=1))
user_bias = tf.get_variable("user_bias", shape=[NB_USERS],
                            initializer=tf.truncated_normal_initializer(stddev=1))
item_bias = tf.get_variable("item_bias", shape=[NB_ITEMS],
                            initializer=tf.truncated_normal_initializer(stddev=1))

user_batch = tf.placeholder(tf.int32, shape=[None])
item_batch = tf.placeholder(tf.int32, shape=[None])
rate_batch = tf.placeholder(tf.float32, shape=[None])

weight_items = tf.nn.embedding_lookup(W_V, item_batch)
weight_users = tf.nn.embedding_lookup(W_U, user_batch)

bias_items = tf.nn.embedding_lookup(item_bias, item_batch)
bias_users = tf.nn.embedding_lookup(user_bias, user_batch)

feat_items = tf.nn.embedding_lookup(B, item_batch)
feat_users = tf.nn.embedding_lookup(A, user_batch)

pred = (tf.reduce_sum(tf.multiply(feat_users, weight_items), 1)
        + tf.reduce_sum(tf.multiply(feat_items, weight_users), 1)
        + bias_items
        + bias_users)
# pred = (tf.reduce_sum(tf.multiply(tf.matmul(feat_users, M), feat_items), 1)
#         + bias_items
#         + bias_users)
cost_l2 = tf.losses.mean_squared_error(rate_batch, pred)

l2_user = tf.nn.l2_loss(weight_users)
l2_item = tf.nn.l2_loss(weight_items)
l2_bias_user = tf.nn.l2_loss(bias_users)
l2_bias_item = tf.nn.l2_loss(bias_items)
regularizer = tf.add(l2_user, l2_item)
regularizer = tf.add(regularizer, l2_bias_user)
regularizer = tf.add(regularizer, l2_bias_item)
# regularizer = tf.nn.l2_loss(M)
penalty = tf.constant(LAMBDA_REG, dtype=tf.float32, shape=[])
cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))

global_step = tf.train.get_global_step()
train_op = tf.train.AdamOptimizer(0.1).minimize(cost, global_step=global_step)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(NB_EPOCHS):
        _, train_pred, train_mse, reg, pen, train_cost = sess.run([train_op, pred, cost_l2, regularizer, penalty, cost], feed_dict={
            user_batch: train['user'],
            item_batch: train['item'],
            rate_batch: train['rating']
        })
        test_pred, test_mse = sess.run([pred, cost_l2], feed_dict={
            user_batch: test['user'],
            item_batch: test['item'],
            rate_batch: test['rating']
        })
        print('train rmse', train_mse ** 0.5, 'test rmse', test_mse ** 0.5)
        # print('reg', reg, 'full cost', train_cost)
