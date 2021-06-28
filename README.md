# Flatland
Generative models (GAN, VAE, DDPM, and Transformers) trained in one and two-dimensional 
data. The point of this is simple: lowering the dimensionality of the training data 
will aid newcomers in the field to better understand the mechanics of what these 
networks do: in particular, what they *learn*. Whenever some work is inspired by 
previous work, it will be duly credited.

For now, this project will be developed using [`PyTorch`](https://pytorch.org/), 
but if possible/if I have both time and patience, I will expand it to other frameworks 
such as `Keras`/`TensorFlow 2.0`, `Julia`, etc. (this is my way of saying I will 
accept pull requests for anyone interested in contributing).

## TODO: 
- [ ] Complete 1D GAN code (the following is not necessarily in order and some 
  may even overlap)
  - [ ] Make `train.py` and `generate.py` code
  - [ ] Add manual seed to latents to easily sample from and create interpolations
    and whatnot (in both scripts)
  - [ ] Add command-line arguments (make most things controllable, but with 
    default values); use [`click`](https://click.palletsprojects.com) for this
  - [ ] Add samples of training results to README (distribution plots, videos)
  - [ ] Make the neural networks easily editable using `config.yml`
- [ ] Ibidem for VAE, but not as pressing
- [ ] Ibidem for DDPM
- [ ] Ibidem for Transformer
- [ ] Add/update `requirements.txt`