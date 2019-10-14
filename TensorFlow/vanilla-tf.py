import tensorflow as tf # tf 1.14 for now
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def glorot_init(size):
    in_dim = size[0]
    out_dim = size[1]

    stddev = tf.sqrt(2. / (in_dim + out_dim))

    return tf.random.truncated_normal(shape=size, stddev=stddev)



# Random noise setting for Generator
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

# Generator parameter settings
G_W1 = tf.Variable()
