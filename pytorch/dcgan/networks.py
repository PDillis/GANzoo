import torch
import torch.nn as nn

## Weight Initialization ##

# All model weights are randomly initialized: N(0, 0.02)

# Custom weights initialization called on netG and netD (need to study this more):
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
