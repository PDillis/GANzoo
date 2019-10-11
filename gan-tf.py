import glob
import os

import imageio
import PIL
import matplotlib.pyplot as plt
from IPython import display

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Load the dataset

(train_images, _), _ = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize images in range [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#### Create the models ####

## GENERATOR ##
