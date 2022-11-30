import torch
import torchvision.transforms as transforms


def get_lenet_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # validation == test
    val_transform = test_transform

    return train_transform, val_transform, test_transform
