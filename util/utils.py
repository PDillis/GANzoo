import sys
import os
from typing import Union, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn

##########################################################################################
#                                   Utility functions
##########################################################################################


def set_device(use_gpu: bool = True) -> torch.device:
    """
    Args:
        use_gpu (bool): whether or not the user wishes to use a GPU
    Outputs:
        device (torch.device): Sets the device and prints it
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f'Current device: {device}')
    return device


def print_stats(use_gpu: bool = True) -> None:
    """
    Args:
        use_gpu (bool): whether or not the user wishes to use GPU
    Outputs:
        (None):
            Prints, for each available GPU, the device name, as well as its compute capability
            If none are available or the user doesn't wish to use one, the CPU will be used and printed accordingly
    """
    print(f'PyTorch version: {torch.__version__}')
    if torch.cuda.is_available() and use_gpu:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            high, low = torch.cuda.get_device_capability(i)
            print(f'GPU {i}, Device: {torch.cuda.get_device_name(i)}, Compute Capability: {high}.{low}')
    else:
        print('Using only CPU...')


def get_latents(num_latents: int, of_type: str,
                device: str = 'cpu',
                latent_dim: int = 5) -> torch.Tensor:
    """
    Generate a set of latent vectors (from Normal or Uniform distribution) to be used by the Generator
    Args:
        num_latents (int): number of latent vectors to generate
        of_type (str): type of latent vectors to generate (choices: 'normal' or 'uniform')
        device (str): device to use; either 'cuda' or 'cpu'
        latent_dim (int): size of the latent dimension Z
    Outputs:
        latents (torch.Tensor): tensor of latent vectors of shape (n, latent_dim)
    """
    dict_dist = {'normal': torch.randn, 'uniform': torch.rand}
    try:
        return dict_dist[of_type](num_latents, latent_dim, device=device)
    except KeyError:
        # TODO: remove this error with click.Choice; make other distributions available?
        print('Please use either "normal" or "uniform" as the distribution of the latent vectors (of_type).')
        sys.exit(1)


def network_summary(network: nn.Module,
                    weights: bool = False,
                    summary: bool = False,
                    input_size: Tuple[int] = (1, )) -> None:
    """
    Print a network summary using torchsummary (model size and trainable/untrainable parameters)
    Args:
        network (nn.Module): neural network that you wish to print its weights/parameters
        weights (bool): Print the values of the weights (recommended only for small networks)
        summary (bool): Whether or not to print a summary of the network
        input_size (tuple): Size of input to simulate a batch that passes through the network
    Output:
        (NoneType): Prints the values of the weights in the network and/or a summary
                (Layers, Output Shape, Param #)
    """
    if weights:
        print('Printing the weights...')
        for i, param in enumerate(network.parameters()):
            print(f'\tLayer {i}:\n{param.data}')
    # If user wishes to print the summary of the network
    if summary:
        try:
            from torchsummary import summary
            summary(model=network, input_size=input_size)
        except ModuleNotFoundError:
            # Won't print the network summary, but also won't quit the program
            print('torchsummary not found! Install it via `pip install torchsummary`')


# init_dict for initializing the neural networks
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


def weights_init(module: nn.Module, init_type: str = 'kaiming', const: float = None, bias_init: float = 0.0) -> None:
    """
    Auxiliary function to initialize the parameters of a network
    Args:
        module (nn.Module): Network to initialize
        init_type (str): type of initialization to apply; must be in init_dict (though I've listed all available inits)
        const (float): constant value to fill the weight, used if init_type='constant'
        bias_init (float): value to initialize the bias (we zero-initialize the bias by default)
    Output:
        (NoneType), applies the desired initialization to the module's layers
    """
    # We will use a set of available initializations
    if init_type not in init_dict:
        print(f'{init_type} not available.')
        sys.exit(1)
    if isinstance(module, nn.Linear):
        # The special case will be the constant initialization
        if init_type == 'constant':
            # Make sure the user has provided the constant value
            # (guard against user forgetting and using a default value)
            assert const is not None, 'Please provide the constant value! (const)'
            # Then, initialize the weight with the provided constant
            init_dict[init_type](module.weight, const)
        else:
            # Else, it's one of the other initialization methods:
            init_dict[init_type](module.weight)
        # Initialize the bias with zeros (in-place); can be changed if so desired
        module.bias.data.fill_(bias_init)


# Global variables; will depend in the end of the actual data generated
actual_mean = 4.0
actual_std = 0.5

def summarize_real_data(data: torch.Tensor) -> None:
    """
    Get the mean and standard deviation of the data; will be saved in the global variables
    Args:
        data (torch.Tensor): real data to summarize/obtain its metrics from
    Returns:
        (NoneType): data mean and standard deviation are saved in the respective global variables
    """
    # We'll use the global actual_mean and actual_std from the real data
    global actual_mean
    global actual_std
    actual_mean = torch.mean(data)
    actual_std = torch.std(data)


def plot_distribution(data: torch.Tensor,
                      fake_data: Union[torch.Tensor, None] = None,
                      epoch: int = 0,
                      hist: bool = False,
                      kde: bool = True,
                      figsize: Union[List[int], Tuple[int]] = (8, 6),
                      root: Union[str, os.PathLike] = os.path.join(os.getcwd(), 'training_runs')) -> None:
    """
    Plot the distribution of both the data and the fake data; the latter can be missing
    Args:
        data (torch.Tensor): real data that is being mimicked
        fake_data ([torch.Tensor, None]): current generated data; can be missing
        epoch (int): current training epoch
        hist (bool): whether or not to plot the histogram of the distribution
        kde (bool): whether or not to add the plot the KDE of the distribution
        figsize ([list, tuple]): size of the figure: (width, height)
        root ([str, os.PathLike]): save root/path where the plot image will be saved to; TODO: create new one per training run
    Output:
        (NoneType) Plot of the real and fake/generated data distributions, saved in the root directory
    """
    fig = plt.figure(figsize=figsize)
    # We plot the generated data distribution if it exists (some params can be modified if desired)
    if fake_data is not None:
        sns.distplot(fake_data, hist=hist, norm_hist=True, bins=50, rug=True,
                     kde=kde, label='Generated data distribution', color='g', rug_kws={'alpha': 0.1})
    # We will compare it to the original/real data distribution
    sns.distplot(data, hist=False, label='Real data distribution', kde_kws={'linestyle': '--', 'color': 'k'})
    plt.title(f'Generated Data Distribution - Epoch {epoch}')
    # The plot limits will be dependent on the real data (i.e., we center around it)
    plt.ylim((0, 1.5))
    plt.xlim((actual_mean - 4.0, actual_mean + 4.0))  # pure heuristics on my part
    # Create the save path (root), if it doesn't exist
    os.makedirs(os.path.join(root, 'training_animation'), exist_ok=True)
    # Save the plot of the distribution at that epoch
    plt.tight_layout()
    save_name = os.path.join(root, 'training_animation', f'g_distr_epoch_{epoch:03d}.png')
    plt.savefig(save_name)
    plt.close(fig)


def plot_losses(
        disc_loss: List[torch.Tensor],
        gen_loss: List[torch.Tensor],
        figsize: Union[List[int], Tuple[int]] = (8, 6)) -> None:
    """
    Plot the losses that were saved during the whole training run.
    TODO: should the losses be saved as tensors? as .pt files?
    Args:
        disc_loss (list): Discriminator loss during training
        gen_loss (list): Generator loss during training
        figsize ([list, tuple]): Figure size
    Output:
        (NoneType) Plot of the losses from the Discriminator and Generator networks, saved in the root directory
    """
    return None


def format_time(seconds: Union[int, float]) -> str:
    """
    Helper function for printing the total training time (TODO: is this even necessary?)
    Args:
        seconds ([int, float]): Seconds that have passed since beginning training
    Output:
        (str): Convert the seconds into a human-readable string
    """
    s = int(np.rint(seconds))
    if s < 60:
        return f'{s}s'
    elif s < 60 * 60:
        return f'{s//60}m {s%60}s'
    elif s < 24 * 60 * 60:
        return f'{s // (60 * 60)}h {(s // 60) % 60}m {s%60}s'
    else:
        return f'{s // (24 * 60 * 60)}d {(s // (60 * 60)) % 24}h {(s // 60) % 60}m'
