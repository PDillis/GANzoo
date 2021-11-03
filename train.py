import click
import torch


##########################################################################################
#                   Main function to tie the training of all the models
##########################################################################################


@click.group()
def main():
    pass


##########################################################################################
#                                       Train a GAN
##########################################################################################


@main.command(name='gan')
@click.pass_context
@click.option('--lr', 'learning_rate', type=float,
              help='Learning rate for both networks (G and D)', default=1e-4, show_default=True)
@click.option('--batch-size', '-bs', type=click.IntRange(min=1),
              help='Batch size to use', default=32, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1),
              help='Number of iterations to train the GAN for (i.e., how many batches it will see)',
              default=100, show_default=True)
@click.option('--dataset', type=click.Option(['normal', 'half-normal', 'uniform', 'laplacian', 'petit-prince']),
              help='The type of the dataset to use for training the GAN.', default='normal', show_default=True)
@click.option('--dataset-size', type=int, help='Size of the dataset', default=30000, show_default=True)
@click.option('--seed', '-s', type=int, help='Seed for random number generation', default=0, show_default=True)
@click.option('--use-gpu', is_flag=True, help='Use a GPU for training, if available')
@click.option('--latent-dim', 'latent_dimension', type=int, help='Size of the latent dimension',
              default=5, show_default=True)
def train_gan(
        ctx: click.Context,
        learning_rate: float,
        batch_size: int,
        iterations: int,
        dataset: str,
        dataset_size: int,
        seed: int,
        use_gpu: bool,
        latent_dimension: int,
):
    """
    Example:
        python train.py gan --lr 1e-2 --batch-size 64 --iterations 1000 --dataset laplacian --dataset-size 10000
    """
    print('Training GAN!')

    # Setup device and seeds
    device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
    torch.random.manual_seed(seed)




##########################################################################################
#                                       Train a VAE
##########################################################################################


@main.command(name='vae')
def train_vae():
    print('Training VAE!')


##########################################################################################
#                                        Train DDPM
##########################################################################################


@main.command(name='ddpm')
def train_ddpm():
    print('Training DDPM!')


##########################################################################################
#                                   Train a Transformer
##########################################################################################


@main.command(name='transformer')
def train_transformer():
    print('Training Transformer!')


# ================================================


if __name__ == '__main__':
    main()

# ================================================
