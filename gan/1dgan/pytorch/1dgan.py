import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
from torch import nn

import shutil
import os
import time
from typing import Union


##########################################################################################
################################### Utility functions ####################################
##########################################################################################

# to become utility.py

def set_device(use_gpu=True):
    '''
    Args:
        use_gpu (Bool): whether the user wishes to use a GPU or not
    Outputs:
        Sets the device and prints it
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f'Current device: {device}')


def print_stats(use_gpu=True):
    '''
    Args:
        use_gpu (Bool): whether the user wishes to use GPU or not
    Outputs:
        Prints stats for the current system:
            For each GPU, if available, the device name, as well as it's compute capability
            In the end, will print if there's a GPU available or if only CPU is used
    '''
    print("PyTorch version: ", torch.__version__)
    if torch.cuda.is_available() and use_gpu:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            high, low = torch.cuda.get_device_capability(i)
            message = f'GPU {i}, Device: {torch.cuda.get_device_name(i)}, Compute Capability: {high}.{low}'
            print(message)


def get_latents(n, of_type, device=device, latent_dim=5):
    '''
    Args:
        n (int): number of latent vectors to generate
        of_type (str): type of latent vectors to generate (choices: 'normal', 'uniform')
        latent_dim (int): size of the latent dimension
    Outputs:
        latents (tensor): tensor of latent vectors of shape (n, latent_dim)
    '''
    # Define the shape of the latent vector
    size = (n, latent_dim)
    # Check which distribution to use
    if of_type == 'normal':
        latents = torch.randn(*size, device=device)
    elif of_type == 'uniform':
        latents = torch.rand(*size, device=device)
    else:
        print('Please use either "normal" or "uniform" as the distribution of the latent vectors (of_type).')
        exit()
    return latents


def network_summary(network, data=False, summary=False, input_size=(1, )):
    '''
    Args:
        network (nn.Module): neural network that you wish to print its weights/parameters
        summary (bool): Whether or not to print a summary of the network
        input_size ()
    Output:
        Prints the values of the weights in the network, and a summary (Layers, Output Shape, Param #)
    '''
    if data: 
        for i, param in enumerate(network.parameters()):
            print(param.data)
    # If user wishes to print the summary of the network
    if summary:
        from torchsummary import summary
        summary(model=network, input_size=input_size)


def weights_init(module, init_type='kaiming', const=None):
    '''
    Args:
        module (nn.Module): layer of a neural network
        init_type (str): type of initialization [kaiming]
        const (float): constant value to fill the weight, if init_type='constant'
    Output:
        None, applies the desired initialization to the layers
    '''
    # We define the dictionary with all the available initializations (except constant)
    init_dict = {
        'uniform': nn.init.uniform_,    # U([0, 1])
        'normal': nn.init.normal_,      # N(0, 1)
        'constant': nn.init.constant_,
        'ones': nn.init.ones_,
        'zeros': nn.init.zeros_,
        'eye': nn.init.eye_,
        'dirac': nn.init.dirac_,
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier': nn.init.xavier_normal_, 
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming': nn.init.kaiming_normal_,
        'orthogonal': nn.init.orthogonal_,
        'sparse': nn.init.sparse_
    }
    if isinstance(module, nn.Linear):
        # The special case will be the constnat initialization
        if init_type=='constant':
            # Make sure the user has provided the constant value
            assert const != None, 'Please provide the constant value! (const)'
            # Then, initialize the weight with the provided constant
            init_dict[init_type](module.weight, const)
        else:
            # Else, it's one of the other initialization methods:
            init_dict[init_type](module.weight)
        # Initialize the bias with zeros (in-place)
        module.bias.data.fill_(0.0)


def plot_distribution(data, 
                      fake_data, 
                      epoch, 
                      hist=False, 
                      kde=True, 
                      figsize=(8, 6), 
                      root='./animation'):
    '''
    Args:
        data (torch.Tensor): real data that is being mimicked
        fake_data (torch.Tensor): current generated data that is trying to mimic data
        epoch (int): current training epoch
        hist (bool): whether or not to plot the histogram of the distribution
        kde (bool): whether or not to plot the KDE of the distribution
        figsize (tuple): size of the figure (width, height)
        root (str): save root/path where the plot image will be saved to
    Output:
        Plot of the real and fake/generated data distributions, saved in the root directory
    '''
    fig = plt.figure(figsize=figsize)
    # We plot the generated data distribution (some values can be modified if desired)
    sns.distplot(fake_data, hist=hist, norm_hist=True, bins=50, rug=True, kde=kde, label='Generated data distribution', color='g', rug_kws={'alpha': 0.1})
    # We will compare it to the original/real data distribution
    sns.distplot(data, hist=False, label='Real data distribution', kde_kws={'linestyle': '--', 'color': 'k'})
    plt.title(f"Generated Data Distribution - Epoch {epoch}")
    # The plot limits will be dependent on the real data (i.e., we center around it)
    plt.ylim((0, 1.5))
    plt.xlim((actual_mean - 4.0, actual_mean + 4.0)) # pure heuristics on my part
    # If the save path (root) doesn't exist, create it
    if not os.path.exists(root):
        os.mkdir(root)
    # Save the plot of the distribution at that epoch
    plt.tight_layout()
    save_name = root + f"/g_distr_epoch_{epoch:03d}.png"
    plt.savefig(save_name)
    plt.close(fig)


def plot_losses(D_loss, G_loss, figsize=(8, 6)):
    


def format_time(seconds: Union[int, float]) -> str:
    '''
    Args:
        seconds ([int, float]): Seconds that have passed
    Output:
        Convert the seconds into a human-readable string
    '''
    s = int(np.rint(seconds))
    if s < 60:
        return f"{s}s"
    elif s < 60 * 60:
        return f"{s//60}m {s%60}s"
    elif s < 24 * 60 * 60:
        return f"{s // (60 * 60)}h {(s // 60) % 60}m {s%60}s"
    else:
        return f"{s // (24 * 60 * 60)}d {(s // (60 * 60)) % 24}h {(s // 60) % 60}m"


##########################################################################################
################################### Thre training data ###################################
##########################################################################################

# to become data.py

class NormalDistribution:
    def __init__(self, loc=4.0, scale=0.5):
        self.loc = torch.tensor([loc])
        self.scale = torch.tensor([scale])

    def sample(self, N, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.normal.Normal(loc=self.loc, scale=self.scale)
        # Sample N numbers from this distribution
        samples = m.sample([N])
        return samples


class LaplaceDistribution:
    def __init__(self, loc=3.0, scale=0.3):
        self.loc = torch.tensor([loc])
        self.scale = torch.tensor([scale])

    def sample(self, N, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.Laplace(loc=self.loc, scale=self.scale)
        # Sample N numbers from this distribution
        samples = m.sample([N])
        return samples


class HalfNormalDistribution:
    def __init__(self, scale=0.75):
        self.scale = torch.tensor([scale])

    def sample(self, N, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.HalfNormal(scale=self.scale)
        # Sample N numbers from this distribution
        samples = m.sample([N])
        return samples


class PetitPrinceDistribution:
    def __init__(self, loc1=4.0, scale1=1.5, loc2=0.6, scale2=1.35):
        self.loc1 = torch.tensor([loc1])
        self.scale1 = torch.tensor([scale1])
        self.loc2 = torch.tensor([loc2])
        self.scale2 = torch.tensor([scale2])

    def sample(self, N, seed=42):
        # Set the seed for reproducibility:
        torch.manual_seed(seed)
        # Define the distributions:
        m1 = torch.distributions.normal.Normal(loc=self.loc1, scale=self.scale1)
        m2 = torch.distributions.normal.Normal(loc=self.loc2, scale=self.scale2)
        # Sample N numbers from this distribution
        samples = torch.cat((m1.sample([N//2]), m2.sample([N-N//2])), 0)
        return samples


##########################################################################################
###################################### The Networks ######################################
##########################################################################################

class Generator(nn.Module):
    def __init__(self, latent_dim=latent_dim, hidden_dim=15, output_size=1):
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
real_label = 1
fake_label = 0

criterion = nn.BCELoss()

##########################################################################################
####################################### Train loop #######################################
##########################################################################################

fixed_latent = get_latents(N=batch_size, latent_dim=latent_dim)

def train(num_epochs=50,
          lr=1e-4,
          num_eval=1,
          data=data,
          hist=True,
          kde=False,
          root='./animation',
          d_repeats=1):
    # Some sanity check:
    wow = "Epochs must be at least 1 and an int!"
    assert num_epochs > 0 and isinstance(num_epochs, int), wow

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
            # Move the batch data to the device:
            real_b = d.to(device)
            # We want the batch size to create the labels and latent vectors
            # later on (remember the last batch won't be of size 1024):
            b_size = real_b.size(0)
            # Label of real data:
            label = torch.full((b_size, ), 
                               real_label, # we fill it with ones
                               device=device)
            # Classify the real batch with D:
            output = discriminator(real_b).view(-1)
            # Calculate the loss on this real batch:
            err_D_real = criterion(output, label)
            # Calculate the gradients for D in backward pass:
            err_D_real.backward()
            # We calculate the average classification of the real data to monitor
            # its progression (should start high, then get lower)
            D_X = output.mean().item()

            ### b. Train D with a fake batch:
            # Generate a batch of latent vectors:
            latents = get_latents(N=b_size, 
                                  latent_dim=latent_dim)
            # Generate fake data with G:
            fake_b = generator(latents).to(device)
            # Let's reuse label (hence in-place fill) by filling with 0's:
            label.fill_(fake_label)
            # Classify the fake batch with D:
            output = discriminator(fake_b.detach()).view(-1)
            # Calculate D's loss on this fake batch:
            err_D_fake = criterion(output, label)
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
            # D d_repeats every time we train G one time
            if i % d_repeats == 0:
                ##############################
                #                   c.
                # Update G: max log(D(G(z)))
                ##############################
                ## c. Train G with a fake batch:
                optimizerG.zero_grad()
                # This time, for the Generator, we fill the label with 1's:
                label.fill_(real_label)
                # We updated D before this, so we make another forward-pass of an 
                # all-fake batch:
                output = discriminator(fake_b).view(-1)
                # Calculate G's loss based on this output:
                err_G = criterion(output, label)
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
                "  Loss_G: %.4f  D(x): %.4f  D(G(z)): %.4f").format(
                    *2*[epoch_digits], *2*[batch_digits])
                # Print it:
                print(message % (epoch, 
                                 num_epochs, 
                                 i, 
                                 len(dataloader),
                                 format_time(runtime), 
                                 err_D.item(), 
                                 err_G.item(),
                                 D_X, 
                                 D_G_z))

        # After every epoch, we save the losses:
        G_losses.append(err_G.item())
        D_losses.append(err_D.item())

        # Check how the generator is doing by saving G's output on the fixed_latent
        with torch.no_grad():
            fake_X = generator(fixed_latent).detach().cpu()
        fake_data.append(fake_X)
        # Let's plot this fake data distribution and save it:
        plot_distribution(data, 
                          fake_X, 
                          epoch, 
                          hist=hist,
                          kde=kde,
                          root=root)
    print("***** Finished training *****")    
    
    # Return both losses for the Generator and Discriminator, as well as the 
    # fake data generated with our fixed_latent
    return G_losses, D_losses, fake_data


G_losses, D_losses, fake_data = train(num_epochs=100, 
                                      lr=1e-4, 
                                      num_eval=10, 
                                      data=data,
                                      hist=False,
                                      kde=True,
                                      root='./animation',
                                      d_repeats=1)

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

!ffmpeg -y -framerate 20 -i ./animation/g_distr_epoch%3d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p training_video.mp4

!ffmpeg -y -i training_video.mp4 training_gif.gif

from IPython.display import display, Image

with open('./training_gif.gif', 'rb') as f:
    display(Image(data=f.read(), format='png'))

torch.save(generator.state_dict(), './trained_g.pth')
torch.save(discriminator.state_dict(), './trained_d.pth')

g = Generator(hidden_dim=15)
g.load_state_dict(torch.load('./trained_g.pth'))

d = Discriminator(hidden_dim=25)
d.load_state_dict(torch.load('./trained_d.pth'))

