import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights


def get_resnet_transforms(cfg):
    weights = ResNet18_Weights.DEFAULT
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    test_transform = weights.transforms()

    # validation == test
    val_transform = test_transform

    return train_transform, val_transform, test_transform

