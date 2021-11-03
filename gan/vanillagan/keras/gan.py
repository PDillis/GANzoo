# From GANs in Action: https://www.manning.com/books/gans-in-action

# Standard imports (for TensorFlow >= 1.14)
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# We will use the MNIST dataset, but in the future we wish to generalize this
# to any image size, so as to not depend on keras datsets, but more on any image
# dataset that the user wishes to use these programs with.

# MNIST are 28x28 images of 1 channel
img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions:
img_shape = (img_rows, img_cols, channels)

# The latent dimension/noise vector shape
z_dim = 100

########## Generator ##########

def build_generator(img_shape, z_dim, alpha=0.01, activation='tanh'):
    """
    Our Generator will only have a single hidden layer, taking z as input and
    producing a 28x28x1 image. We will use LeakyReLU, tanh as the output activation
    function, as, apparently, tanh produces crisper images, but this is something
    that for sure can be easily tested.

    input: latent vector of shape (z_dim, )
    oputut: 28x28x1 image
    """
    model = Sequential()
    # The FC layer:
    model.add(Dense(128, input_dim=z_dim))
    # The LeakyReLU activation:
    model.add(LeakyReLU(alpha=alpha))
    # Output layer with tanh activation:
    model.add(Dense(img_rows*img_cols*channels, activation=activation))
    # Reshape the Generator output to image dimensions:
    model.add(Reshape(img_shape))

    return model

########## Discriminator ##########

def build_discriminator(img_shape, alpha=0.01):
    """
    Our Discriminator takes a 28x28x1 image and outputs a probability [0, 1], with
    1 indicating that D believes the input image to be 'real' and 0 otherwise. It
    will have a hidden layer with 128 hidden units and LeakyReLU. For the output,
    however, we will use sigmoid.

    input: 28x28x1 image
    output: P(image being real|dataset) (real number between 0 and 1)
    """
    model = Sequential()
    # We flatten the image:
    model.add(Flatten(input_shape=img_shape))
    # The FC layer:
    model.add(Dense(128))
    # The LeakyReLU activation"
    model.add(LeakyReLU(alpha=alpha))
    # Output layer with the sigmoid activation:
    model.add(Dense(1, activation='sigmoid'))
    return model

########## Building the model ##########

def build_gan(generator, discriminator):
    """
    When training the Generator, we will keep the Discriminator parameters fixed
    (discriminator.trainable=False). This model will only be used to train the Generator.
    """
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# As a loss function, we will use the binary crossentropy, since we only have two
# posible classes: real and fake. We will use Adam as our gradient optimizer

# Build and compile the Discriminator:
discriminator = build_discriminator(img_shape=img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build the Generator:
generator = build_generator(img_shape=img_shape, z_dim=z_dim)

# Keep the parameters of the Discriminator constant whilst the Generator trains:
discriminator.trainable = False

# Build and compile the GAN with fixed Discriminator to train the Generator:
gan = build_gan(generator=generator, discriminator=discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

########## Training ##########

# We will get a random minibatch of images as real examples and generate a
# minibatch of fake images from random noise vectors z. We then use those to
# train the Discriminator network while keeping the Generator's parameters
# constant.

# We then generate a minibatch of fake images and use those to train the Generator
# whilst keeping the Discriminator's parameters constant. This is repeated for
# each iteration (e.g., each minibatch or epoch).

# Our labels will simply be one-hot encoded: 1 for real, 0 for fake. The latent
# vectors will be drawn from the Normal distribution: z ~ N(0, 1). Since the
# Generator uses the tanh activation function, we will rescale our images to be
# in the (-1, 1) scale (same for the Discriminator's input).

losses = []
accuracies = []
epoch_checkpoints = []

def train(epochs, batch_size, sample_interval):
    # Load the MNIST dataset:
    (X_train, _), (_, _) = mnist.load_data()
    # Rescale [0, 255] grayscale pixel values to [-1, 1]:
    X_train = X_train / 127.5 - 1.0

    X_train = np.expand_dims(X_train, axis=3)
    # Labels for real images:
    real_labels = np.ones((batch_size, 1))
    # Labels for fake images:
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        # Generate a batch of fake images:
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        # Train the Discriminator on the batch of real and fake images
        d_loss_real = discriminator.train_on_batch(imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the Generator:
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, real_labels)
        # Every sample_interval, do the following:
        if (epoch + 1) % sample_interval == 0:
            # Save losses and accuracies to be plotted afterwards:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            epoch_checkpoints.append(epoch + 1)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch + 1,
                                                                  d_loss,
                                                                  100.0*accuracy,
                                                                  g_loss))
            # Sample images:
            sample_images(generator)


# Now, we must define this sample_images function:
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    """
    This function will generate a 4x4 (default) grid of images generated by the
    Generator, which will be run every sample_interval. Useful for evaluating
    our model and to early stop it.
    """
    # Sample random noise:
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    # Generate the images from this noise:
    gen_imgs = generator.predict(z)
    # Rescale image pixel values to [0, 1]:
    gen_imgs = 0.5 * (gen_imgs + 1)
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

# Set hyperparameters:
epochs = 2000
batch_size = 128
sample_interval = 100

train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
