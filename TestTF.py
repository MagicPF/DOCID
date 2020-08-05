import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g = tf.Graph()
with g.as_default():
    # one = tf.constant([[3,3]])
    # two = tf.constant([[2],[2]])
    # p = tf.matmul(one,two)
    sess = tf.Session(graph=g)
    # re=sess.run(p)
    hello = tf.constant('Hello TensorFlow')
    print(sess.run(hello).decode())

