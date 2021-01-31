import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
from torch import nn

import shutil
import time

from util.data import NormalDistribution, LaplaceDistribution, HalfNormalDistribution, PetitPrinceDistribution
from util.utils import set_device, print_stats, get_latents, network_summary, weights_init, plot_distribution, \
    plot_losses, format_time

##########################################################################################
###################################### The Networks ######################################
##########################################################################################

class Generator(nn.Module):
    def __init__(self, latent_dim=5, hidden_dim=15, output_size=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is the latent vector with ReLU output to the first hidden layer
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            # The hidden layer with linear output
            nn.Linear(in_features=hidden_dim, out_features=output_size)
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_size=1, hidden_dim=25):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is one-dimensional
            nn.Linear(in_features=input_size, out_features=hidden_dim),
            # We use ReLU Non-linearity
            nn.ReLU(inplace=True),
            # Our output is a probability, so we use one output neuron with Sigmoid
            nn.Linear(in_features=hidden_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


##########################################################################################
####################################### Parameters #######################################
##########################################################################################

# to become config.yml
batch_size = 512
latent_dim = 5
lr_gen = 0.0001
lr_dis = 0.0001

device = set_device(use_gpu=True)

criterion = nn.BCELoss()

##########################################################################################
####################################### Train loop #######################################
##########################################################################################

generator = Generator().to(device)
discriminator = Discriminator().to(device)

fixed_latent = get_latents(num_latents=batch_size, latent_dim=latent_dim)

def train(discriminator, generator, num_epochs=50, lr=1e-4, num_eval=1, data=data,
          hist=True, kde=False, root='./animation', d_repeats=1):
    # Some sanity check:
    assert num_epochs > 0 and isinstance(num_epochs, int), "Epochs must be at least 1 and an int!"
    # We will use Adam for both optimizers with same learning rate:
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)

    # We will keep track of the generator and discriminator losses:
    G_losses = []
    D_losses = []

    # We will log the generated fake data (with the fixed_latent) in order to 
    # track our progression throughout training (i.e., plot it later on):
    fake_data = []

    # This will remove the previous run subdir (should this be the second time 
    # you run this code; make sure to save the previous run if you so wish):
    shutil.rmtree(root, ignore_errors=True)

    # Mark the beginning of training time:
    start_time = time.time()

    print("***** Starting training *****")
    for epoch in range(num_epochs):
        # For each batch in our dataloader:
        for i, d in enumerate(dataloader):
            ############################################
            #                  a.              b.
            # Update D: max log(D(x)) + log(1 - D(G(z)))
            ############################################
            ### a. Train D with a real batch. First zero_grad the optimizer:
            optimizerD.zero_grad()
            # Move the batch of data to the device:
            real_b = d.to(device)
            # We want the batch size to create the labels and latent vectors
            # later on (remember the last batch won't be of size 1024):
            batch_size = real_b.size(0)
            # Label of real data:
            real_label = torch.ones_like(real_b).to(device)
            # Classify the real batch with D:
            output = discriminator(real_b).view(-1)
            # Calculate the loss on this real batch:
            err_D_real = criterion(output, real_label)
            # Calculate the gradients for D in backward pass:
            err_D_real.backward()
            # We calculate the average classification of the real data to monitor
            # its progression (should start high, then get lower)
            D_X = output.mean().item()

            ### b. Train D with a fake batch:
            # Generate a batch of latent vectors:
            latents = get_latents(n=batch_size, latent_dim=latent_dim)
            # Generate fake data with G:
            fake_batch = generator(latents).to(device)
            # Label of fake data:
            fake_label = torch.zeros_like(fake_batch).to(device)
            # Classify the fake batch with D (we must detach it from the Generator):
            output = discriminator(fake_batch.detach()).view(-1)
            # Calculate D's loss on this fake batch:
            err_D_fake = criterion(output, fake_label)
            # Calculate the gradients for D in backward pass:
            err_D_fake.backward()
            # We calculate the average classification of the fake data to monitor
            # its progression:
            D_G_z = output.mean().item()

            # Add both gradients from the all-real and all-fake batches:
            err_D = err_D_real + err_D_fake
            # Once we've accumulated both gradients in the backward pass, we take
            # a step:
            optimizerD.step()

            # Now on to train the Generator: remember we are in essence training
            # the Discriminator d_repeats every time we train the Generator one time
            if i % d_repeats == 0:
                ##############################
                #                   c.
                # Update G: max log(D(G(z)))
                ##############################
                ## c. Train G with a fake batch:
                optimizerG.zero_grad()
                # We updated D before this, so we make another forward-pass of an 
                # all-fake batch:
                output = discriminator(fake_batch).view(-1)
                # Calculate G's loss based on this output, now with a 'real' label:
                err_G = criterion(output, real_label)
                # Calculate the gradient for G:
                err_G.backward()
                # Update G:
                optimizerG.step()

            #######################################################################
            # That's it for Algorithm 2; now on to printing some summary statistics
            #######################################################################

            # I am an order maniac, so I wish to print the number of necessary
            # spaces depending on the number of digits for epochs and batches:
            epoch_digits = int(np.log10(num_epochs)) + 1
            batch_digits = int(np.log10(len(dataloader))) + 1
            # Print our training statistics every num_eval:
            if i % num_eval == 0:
                # Get the current time:
                runtime = time.time() - start_time
                # Our code will run in less than 1 hour, so we really only need 
                # space for 6 strings for the runtime (hence %-6s):
                message = ("[%{}d/%{}d][%{}d/%{}d]  Runtime: %-6s  Loss_D: %.4f"
                "  Loss_G: %.4f  D(x): %.4f  D(G(z)): %.4f").format(*2*[epoch_digits], *2*[batch_digits])
                # Print it:
                print(message % (epoch, num_epochs, i, len(dataloader), format_time(runtime),
                                 err_D.item(), err_G.item(), D_X, D_G_z))

        # After every epoch, we save the losses:
        G_losses.append(err_G.item())
        D_losses.append(err_D.item())

        # Check how the generator is doing by saving G's output on the fixed_latent
        with torch.no_grad():  # (without computing the gradients)
            fake_X = generator(fixed_latent).detach().cpu()
        fake_data.append(fake_X)
        # Let's plot this fake data distribution and save it:
        plot_distribution(data, fake_X, epoch, hist=hist, kde=kde, root=root)
    print("***** Finished training *****")    
    
    # Return both losses for the Generator and Discriminator, as well as the 
    # fake data generated with our fixed_latent
    return G_losses, D_losses, fake_data


G_losses, D_losses, fake_data = train(num_epochs=100, lr=1e-4, num_eval=10, data=data,
                                      hist=False, kde=True, root='./animation', d_repeats=1)

means = [d.mean() for d in fake_data]
stds = [d.std() for d in fake_data]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, 0, 1, 1])
plt.plot(means, label='Generated mean')
ax.axhline(y=actual_mean, c='blue', ls='--', label=f'Actual mean: {actual_mean:.3f}')
plt.plot(stds, label='Generated standard dev')
ax.axhline(y=actual_std, c='orange', ls='--', label=f'Actual std dev: {actual_std:.3f}')
plt.xlabel("Epoch")
plt.title(f"Final mean: {means[-1]:.3f}, Final std dev: {stds[-1]:.3f}")
plt.legend()
plt.show()


# Use ffmpeg-python: https://github.com/kkroening/ffmpeg-python
# Or just plain ffmpeg: https://www.wikihow.com/Install-FFmpeg-on-Windows

# !ffmpeg -y -framerate 20 -i ./animation/g_distr_epoch%3d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p training_video.mp4

# !ffmpeg -y -i training_video.mp4 training_gif.gif

from IPython.display import display, Image

with open('./training_gif.gif', 'rb') as f:
    display(Image(data=f.read(), format='png'))

torch.save(generator.state_dict(), './trained_g.pth')
torch.save(discriminator.state_dict(), './trained_d.pth')

g = Generator(hidden_dim=15)
g.load_state_dict(torch.load('./trained_g.pth'))

d = Discriminator(hidden_dim=25)
d.load_state_dict(torch.load('./trained_d.pth'))

if __name__=='__main__':
    pass
