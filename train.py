import click


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
@click.pass_context()
@click.option('--lr', 'learning_rate', type=float, help='Learning rate for both networks (G and D)', default=1e-4, show_default=True)
@click.option('--batch-size', '-bs', type=click.IntRange(min=1), help='Batch size to use', default=32, show_default=True)
@click.option('--')
def train_gan(
        ctx: click.Context,
        learning_rate: float,
        batch_size: int,
):
    """
    Example:
        python train.py gan --lr 1e-2 --batch-size 64
    Returns:

    """
    pass


##########################################################################################
#                                       Train a VAE
##########################################################################################


@main.command(name='vae')
def train_vae():
    pass


##########################################################################################
#                                        Train DDPM
##########################################################################################

@main.command(name='ddpm')
def train_ddpm():
    pass


##########################################################################################
#                                   Train a Transformer
##########################################################################################


@main.command(name='transformer')
def train_transformer():
    pass


# ================================================


if __name__ == '__main__':
    main()

# ================================================
