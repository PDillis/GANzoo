# Flatland
Generative models (GAN, VAE, and Transformers) trained in one and two-dimensional data. 
Lowering the dimensions of the data aid newcomers in the field to better understand 
the mechanics of what these networks do: in particular, what they *learn*. Whenever 
some work is inspired by previous work, it will be duly credited.

For now, this project will be developed using [`PyTorch`](https://pytorch.org/), but if possible 
(in particular, if I have both time and patience), I will expand it to other frameworks, 
such as Keras/TensorFlow, Jax, etc.

## TODO: 
- [ ] Complete 1D GAN code
  - [ ] Add manual seed to latents
  - [ ] Make it easy to sample or generate new fake data with seed (e.g., `generate.py`)
  - [ ] Add command line arguments (make everything controllable, but with default values)
  - [ ] Add samples of training results (distribution plots, videos)
  - [ ] Make the neural networks easily editable using `config.yml`
- [ ] Ibidem for VAE, but not as pressing
- [ ] Add `requirements.txt`