# From GANs in Action: https://www.manning.com/books/gans-in-action

# Standard imports (for tensorflow >= 1.14)
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.datasets import mnist

import numpy as np
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

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

# Access a specific layer:
vae.get_layer('encoder').output

# So encoder outputs [(None, 2), (None, 2), (None, 2)], the first is the mean,
# second is the variance, third is the Lambda function defined above, which will
# return Z, or the sampling.

### Loss function ###
def vae_loss(x, x_decoded_mean, z_mean=z_mean, z_log_var=z_log_var, original_dim=original_dim):
    # Binary crossentropy loss:
    xent_loss = original_dim * losses.binary_crossentropy(x, x_decoded_mean)
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

### Generate new data ###

len(encoder.predict(x_test, batch_size=100))

decoder.predict(encoder.predict(x_test, batch_size=batch_size)).shape

plt.imshow(decoder.predict(encoder.predict(x_test, batch_size=batch_size))[0].reshape(28, 28))

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
# We will then plot just the mean z of each x_test_encoded

# Figure 2.6
plt.figure(figsize=(10, 8))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='viridis')
plt.colorbar()
plt.show()


# Figure 2.7

# Display a 2d manifold of the digits
n = 15 # (15x15 images)
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# Since the latent space is Gaussian:
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xj in enumerate(grid_y):
        z_sample = np.array([[xj, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit


plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()


x_test_encoded.shape
len(encoder.predict(x_test, batch_size=batch_size)[2])

x_test_encoded[0]

x_test_encoded_decoded = decoder.predict(x_test_encoded)


def mse(x, y):
    return np.linalg.norm(x - y)

ssim_array = np.array([ssim(x_test[i].reshape(28, 28), x_test_encoded_decoded[i].reshape(28, 28), full=True)[1] for i in range(columns)])

images = np.vstack((x_test[:columns], x_test_encoded_decoded[:columns], ssim_array.reshape(columns, -1)))

def plot(images, columns=10, rows=3):
    fig=plt.figure(figsize=(2*columns, 2*rows))
    for i in range(columns*rows):
        img = images[i]
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='Greys_r')
        if i == 0:
            plt.ylabel("OG image")
        if i == columns:
            plt.ylabel("Rec. image")
        if i == 2 * columns:
            plt.ylabel("SSIM")
        if i >= (rows - 1) * columns:
            s_sim = ssim(images[i - 2 * columns], images[i - columns])
            ms = mse(images[i - 2 * columns], images[i - columns])
            xlabel = "SSIM={:.3f}\nMSE={:.3f}".format(s_sim, ms)
            plt.xlabel(xlabel)
        plt.xticks([])
        plt.yticks([])
    plt.show()

plot(images)

ssim(images[0].reshape(28, 28), images[10].reshape(28, 28))


#################### Second try, with different loss ####################


### Loss function ### (this is incorrect)
def ssim_loss(x, x_decoded_mean, z_mean=z_mean, z_log_var=z_log_var, original_dim=original_dim):
    # SSIM loss:
    ssim_loss = ssim(x, decoder.predict(encoder.predict(x)))
    # KL divergence:
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return ssim_loss + kl_loss
