import torch.nn as nn

"""
Separate code as such:
    load the dataset with the respective dataloader (one file)
    for number of iterations/batches:
        train the model with a function handling backprop and step (this file)
        using another code will have the available losses to be selected
        
"""


def train_for_one_epoch(model):
    pass