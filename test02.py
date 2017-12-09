# coding=utf-8
import tensorflow as tf
import numpy as np

a = [[1, 2, 3], [4, 5, 6]]
anp = np.array(a)
print(anp.shape)

cluster = tf.train.ClusterSpec({
    "worker": [
        "worker_task0.example.com:2222",  # /job:worker/task:0 运行的主机
        "worker_task1.example.com:2222",  # /job:worker/task:1 运行的主机
        "worker_task2.example.com:2222"  # /job:worker/task:3 运行的主机
    ],
    "ps": [
        "ps_task0.example.com:2222",  # /job:ps/task:0 运行的主机
        "ps_task1.example.com:2222"  # /job:ps/task:0 运行的主机
    ]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

input = tf.placeholder(dtype=tf.int32, shape=[2])
label = tf.placeholder(dtype=tf.int32, shape=[2])

with tf.device("/job:ps/task:0"):
    weights_1 = tf.Variable(tf.random_normal([2, 2]))
    biases_1 = tf.Variable(tf.constant(0.1, shape=[2, ]))

with tf.device("/job:ps/task:1"):
    weights_2 = tf.Variable(tf.random_normal([2, 2]))
    biases_2 = tf.Variable(tf.constant(0.1, shape=[2, ]))

with tf.device("/job:worker/task:0"):  # 映射到主机(10.1.1.1)上去执行

    layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
    logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
with tf.device("/job:worker/task:1"):  # 映射到主机(10.1.1.2)上去执行

    layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
    logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
    # ...
    train_op = ...
with tf.Session("grpc://10.1.1.2:3333") as sess:  # 在主机(10.1.1.2)上执行run
    for _ in range(10000):
        sess.run(train_op)



inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None])
result = tf.reverse(inputs, [-1])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(result, feed_dict={inputs: a}))
