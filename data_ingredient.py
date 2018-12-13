"""Ingredient for making a model with a wrapped lstm and a
head for hidden state, using the CharDecoderHead."""

import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Grayscale
from sacred import Ingredient

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def data_config():
    """Config for data source and loading"""
    batch_size = 32
    device = 'cpu'
    val_split = 0.05
    num_workers = 0 # number of subprocesses apart from main for data loading

@data_ingredient.capture
def make_dataloaders(batch_size,
                     num_workers,
                     val_split,
                     device,
                     _log):
    """Make the required DataLoaders and datasets.

    Parameters
    ----------
    batch_size : int
        batch_size for DataLoader, default is 32.
    num_workers : int
        num_workers for DataLoader, default is 0.
    val_split : float
        ratio of dataset used for validation, default is 0.01.
    device : str
        device to load the DataLoader
    _log : logger
        logger instance

    Returns
    -------
    tuple: dataset, train_loader, val_loader, test_loader
        Returns the dataset, and the train, validation, and test DataLoaders.

    """

    to_device = lambda x:x.to(device)
    dset = MNIST("data", download=True,
                 transform=Compose([Grayscale(), ToTensor(), to_device]),
                 target_transform=to_device)

    test_dset = MNIST("data", download=True, train=False,
                      transform=Compose([Grayscale(), ToTensor(), to_device]),
                      target_transform=to_device)

    _log.info("Loaded dataset")

    total = len(dset)
    train_num = int(total*(1-val_split))
    val_num = total-train_num

    _log.info(f"Split dataset into {train_num} train samples and {val_num} \
    validation samples")

    train, val = torch.utils.data.dataset.random_split(dset,
                                                       [train_num, val_num])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)

    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)


    return dset, train_loader, val_loader, test_loader
