# Standard imports (for tensorflow >= 1.14)
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import objectives
from tensorflow.keras.datasets import mnist
import numpy as np


# Set global variables and hyperparameters

batch_size = 100
original_dim = 28*28
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5
epsilon_std = 1.0

### Creating the Encoder ###

# Input to encoder
x = Input(shape=(original_dim, ), name='input')
# Intermediate layer
h = Dense(intermediate_dim, activation='relu', name='encoding')(x)
# Mean and log-variance of the latent space
z_mean = Dense(latent_dim, name='mean')(h)
z_log_var = Dense(latent_dim, name='log-variance')(h)
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])
# We define the encoder (# Model(start, [end])):
encoder = Model(x, [z_mean, z_log_var, z], name='encoder')

### Decoder ###

# Now we must sample from the latent space and feed this to the Decoder
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# The decoder input:
input_decoder = Input(shape=(latent_dim, ), name='decoder_input')
# Take the latent space to the intermediate dimension:
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')(input_decoder)
# Get the mean from the original dimension
x_decoded = Dense(original_dim, activation='sigmoid', name='flat_decoded')(decoder_h)
# Define the decoder as a Keras model:
decoder = Model(input_decoder, x_decoded, name='decoder')

### VAE Model ###
# Combine Encoder and Decoder into a VAE model:
output_combined = decoder(encoder(x)[2]) # we need the 3rd element (z)
# Links the input and overall output
vae = Model(x, output_combined)
# Print the model:
vae.summary()

### Loss function ###
def vae_loss(x, x_decoded_mean, z_mean, z_log_var, original_dim=original_dim):
    # Binary crossentropy loss:
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    # KL divergence:
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# Compile the model:
vae.compile(optimizer='rmsprop', loss=vae_loss)

### Model training ###
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# We first normalize the data and then reshape the images to not be a matrix,
# but instead be a long vector (784 long):
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.astype('float32') / 255
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# We fit and shuffle and monitor the progress with the validation data:
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        verbose=1)
