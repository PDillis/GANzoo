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
        use_gpu (bool): whether the user wishes to use GPU or not
    Outputs:
        (None):
            Prints, for each available GPU, the device name, as well as its compute capability
            If none are available or the user doesn't wish to use one, the CPU will be used and printed accordingly
    """
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available() and use_gpu:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            high, low = torch.cuda.get_device_capability(i)
            print(f'GPU {i}, Device: {torch.cuda.get_device_name(i)}, Compute Capability: {high}.{low}')
    else:
        print('Using only CPU...')


def get_latents(num_latents: int, of_type: str,
                device: torch.device = torch.device('cuda'),
                latent_dim: int = 5) -> torch.Tensor:
    """
    Args:
        num_latents (int): number of latent vectors to generate
        of_type (str): type of latent vectors to generate (choices: 'normal', 'uniform')
        device (torch.device): device to use; either 'cuda' or 'cpu'
        latent_dim (int): size of the latent dimension
    Outputs:
        latents (torch.Tensor): tensor of latent vectors of shape (n, latent_dim)
    """
    dict_dist = {'normal': torch.randn, 'uniform': torch.rand}
    try:
        latents = dict_dist[of_type](num_latents, latent_dim, device=device)
        return latents
    except KeyError:
        print('Please use either "normal" or "uniform" as the distribution of the latent vectors (of_type).')
        sys.exit(1)


def network_summary(network: nn.Module,
                    weights: bool = False,
                    summary: bool = False,
                    input_size: Tuple[int] = (1, )) -> None:
    """
    Args:
        network (nn.Module): neural network that you wish to print its weights/parameters
        weights (bool): Print the values of the weights (might be a long output to the console)
        summary (bool): Whether or not to print a summary of the network
        input_size (tuple): Size of input to simulate a batch that passes through the network
    Output:
        (NoneType): Prints the values of the weights in the network and/or a summary
                (Layers, Output Shape, Param #)
    """
    if weights:
        print("Printing the weights...")
        for i, param in enumerate(network.parameters()):
            print(f"\tLayer {i}:\n{param.data}")
    # If user wishes to print the summary of the network
    if summary:
        from torchsummary import summary
        summary(model=network, input_size=input_size)


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


def weights_init(module: nn.Module, init_type: str = 'kaiming', const: float = None) -> None:
    """
    Args:
        module (nn.Module): layer of a neural network
        init_type (str): type of initialization to apply; must be in  init_dict
        const (float): constant value to fill the weight, if init_type='constant'
    Output:
        (NoneType), applies the desired initialization to the module's layers
    """
    # We will use a set of available initializations
    if init_type not in init_dict:
        print(f"{init_type} not available.")
        sys.exit(1)
    if isinstance(module, nn.Linear):
        # The special case will be the constant initialization
        if init_type == 'constant':
            # Make sure the user has provided the constant value
            # (guard against user forgetting and using a default value set by me, say 0.5)
            assert const is not None, 'Please provide the constant value! (const)'
            # Then, initialize the weight with the provided constant
            init_dict[init_type](module.weight, const)
        else:
            # Else, it's one of the other initialization methods:
            init_dict[init_type](module.weight)
        # Initialize the bias with zeros (in-place)
        module.bias.data.fill_(0.0)


def plot_distribution(data: torch.Tensor,
                      fake_data: torch.Tensor,
                      epoch: int,
                      hist: bool = False,
                      kde: bool = True,
                      figsize: Tuple[int] = (8, 6),
                      root: Union[str, os.PathLike] = os.path.join(os.getcwd(), 'animation')) -> None:
    """
    Args:
        data (torch.Tensor): real data that is being mimicked
        fake_data (torch.Tensor): current generated data
        epoch (int): current training epoch
        hist (bool): whether or not to plot the histogram of the distribution
        kde (bool): whether or not to add the plot the KDE of the distribution
        figsize (tuple): size of the figure (width, height)
        root ([str, os.PathLike]): save root/path where the plot image will be saved to
    Output:
        (NoneType) Plot of the real and fake/generated data distributions, saved in the root directory
    """
    fig = plt.figure(figsize=figsize)
    # We plot the generated data distribution (some values can be modified if desired)
    sns.distplot(fake_data, hist=hist, norm_hist=True, bins=50, rug=True,
                 kde=kde, label='Generated data distribution', color='g', rug_kws={'alpha': 0.1})
    # We will compare it to the original/real data distribution
    sns.distplot(data, hist=False, label='Real data distribution', kde_kws={'linestyle': '--', 'color': 'k'})
    plt.title(f"Generated Data Distribution - Epoch {epoch}")
    # The plot limits will be dependent on the real data (i.e., we center around it)
    actual_mean = torch.mean(data)
    actual_std = torch.std(data)  # TODO: These are both global attributes, where to put them?
    plt.ylim((0, 1.5))
    plt.xlim((actual_mean - 4.0, actual_mean + 4.0)) # pure heuristics on my part
    # If the save path (root) doesn't exist, create it
    if not os.path.exists(root):
        os.mkdir(root)
    # Save the plot of the distribution at that epoch
    plt.tight_layout()
    save_name = os.path.join(root, "training", f"g_distr_epoch_{epoch:03d}.png")
    plt.savefig(save_name)
    plt.close(fig)


def plot_losses(D_loss: List[torch.Tensor], G_loss: List[torch.Tensor], figsize: Tuple[int] = (8, 6)):
    """

    Args:
        D_loss: Discriminator loss during training
        G_loss: Generator loss during trianing
        figsize: Figure size

    Returns:

    """
    return None


def format_time(seconds: Union[int, float]) -> str:
    """
    Helper function for printing the time during training
    Args:
        seconds ([int, float]): Seconds that have passed
    Output:
        (str): Convert the seconds into a human-readable string
    """
    s = int(np.rint(seconds))
    if s < 60:
        return f"{s}s"
    elif s < 60 * 60:
        return f"{s//60}m {s%60}s"
    elif s < 24 * 60 * 60:
        return f"{s // (60 * 60)}h {(s // 60) % 60}m {s%60}s"
    else:
        return f"{s // (24 * 60 * 60)}d {(s // (60 * 60)) % 24}h {(s // 60) % 60}m"
