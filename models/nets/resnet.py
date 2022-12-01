import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet(num_classes, cfg):
    # if pretrain: set weights to pretrained weights
    net = resnet18(weights=ResNet18_Weights.DEFAULT if cfg.pretrain else None)

    # if frozen: only train feature extractor
    if cfg.frozen:
        for params in net.parameters():
            params.requires_grad = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    return net

