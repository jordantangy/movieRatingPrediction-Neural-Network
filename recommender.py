import numpy as np
import tensorflow as tf
import math
import pandas as pd

#retrieve data for specific user(userId, movieId,rating) and separating training set and test set

df = pd.read_csv('ratings.csv', usecols = ['userId','movieId','rating'], low_memory = False)
temp = df.to_numpy()
temp_mat = []
np.set_printoptions(suppress=True)
counter = 0
for row in temp:
    if(row[0]==1):
        temp_mat.append(row)
user_mat = np.array(temp_mat)
user_mat_len = len(user_mat)
training_len = math.floor((user_mat_len*80)/100)
test_len = user_mat_len - training_len



temp_train = []
x = range(0,training_len)
for i in x:
    row = user_mat[i]
    temp_train.append(row)
training_data = np.array(temp_train)



temp_test = []
y = range(training_len,user_mat_len)
for i in y:
    row = user_mat[i]
    temp_test.append(row)
test_data = np.array(temp_test)

# creating neural network
print("started")
features = 1128
df = pd.read_csv('genome-scores.csv', usecols=['movieId', 'relevance'], low_memory=False)
relevance = df.to_numpy()
x = tf.placeholder(tf.float32, shape=[1, features])
y_ = tf.placeholder(tf.float32, [1])
W = tf.Variable(tf.zeros([features, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.pow(y - y_, 2))
update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(len(training_data)):
    movieId = training_data[i][1]
    data_y = [training_data[i][2]]
    temp_mat = []
    np.set_printoptions(suppress=True)
    for j in range(len(relevance)):
        if (relevance[j][0] == movieId):
            temp_mat.append(relevance[j][1])
    data_x = [np.array(temp_mat)]
    print(data_x)
    for k in range(0, 1000000):
        sess.run(update, feed_dict={x: data_x, y_: data_y})
        if (k % 10000 == 0):
            print('Iteration:', k, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
                  loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))
            print("prediction:", np.matmul(data_x, sess.run(W)) + sess.run(b))

