import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm


def online_mean_and_std(loader):
    """
    Computes the mean and std in an online fashion of a loader. Remember:
        -> Mean[x] = E[x] = fst_moment
        -> Var[x] = E[x^2] - E[x]^2 = snd_moment - fst_moment^2

    Input:  type(loader) = <class 'torch.utils.data.dataloader.DataLoader'>
    Output: mean, std: tensors of size 3
    """
    count = 0

    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tqdm(loader, desc="Calculating mean and std; batch"):
        b, c, h, w = images.shape  # [batch, ch, height, width]
        # The following are performed per each channel (3 in total)
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (count * fst_moment + sum_) / (count + nb_pixels)
        snd_moment = (count * snd_moment + sum_of_square) / (count + nb_pixels)
        count += nb_pixels
    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


# Thanks to:
# https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9?u=shubhajitml
def normalized_dataloader_from_ImageFolder(
    dataroot, batch_size, shuffle=True, num_workers=2
):
    """
    Generate the dataloader from torchvision.datasets.ImageFolder(path). For now,
    it will normalize the dataset by finding its mean and std
    """
    # First, we load the dataset as a tensor:
    dataset = dsets.ImageFolder(
        root=dataroot, transform=transforms.Compose([transforms.ToTensor()])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    # Then, we obtain the mean and std per channel:
    data_mean, data_std = online_mean_and_std(dataloader)
    # Then we use these in our dataset:
    dataset = dsets.ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(data_mean, data_std)]
        ),
    )
    # Finally, we have the dataloader:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def device(ngpu):
    if torch.cuda.is_available() and ngpu > 0:
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    return dev


def plot_batch(dataloader, size=8):
    """
    Plot a size x size image grid
    """
    real_batch = next(iter(dataloader))
    assert len(real_batch) >= size * size, "Grid cannot be bigger than a batch"
    plt.figure(figsize=(size, size))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[: size * size], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )
