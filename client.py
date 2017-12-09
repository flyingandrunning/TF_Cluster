import tensorflow as tf
import numpy as np

train_X = np.linspace(-1, 1, 101)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

X = tf.placeholder("float")
Y = tf.placeholder("float")

# 映射到主机(192.168.199.205)上去执行(把变量保存到主机192.168.199.205上)
with tf.device("/job:ps/task:0"):
    w = tf.Variable(0.0, name="wight")
    b = tf.Variable(0.0, name="reminder")

# 映射到主机(192.168.199.167)上去执行
with tf.device("/job:worker/task:0"):
    cost_op = tf.square(Y - tf.multiply(X, w) - b)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost_op)

# 映射到主机(192.168.199.133)上去执行
with tf.device("/job:worker/task:1"):
    cost_op_2 = tf.square(Y - tf.multiply(X, w) - b)
    train_op_2 = tf.train.GradientDescentOptimizer(0.01).minimize(cost_op_2)

print("1")

init_op = tf.global_variables_initializer()
with tf.Session("grpc://192.168.199.167:33333") as sess:
    sess.run(init_op)

    print(2)

    for i in range(5):
        print("here")
        for (x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X: x, Y: y})
            sess.run(train_op_2, feed_dict={X: x, Y: y})

    print(sess.run(w))
    print(sess.run(b))

    # with tf.device("/job:worker/task:1"):
    #     sess.run(init_op)
    #
    #     for i in range(10):
    #         for (x, y) in zip(train_X, train_Y):
    #             sess.run(train_op, feed_dict={X: x, Y: y})
    #     print(sess.run(w))
    #     print(sess.run(b))
