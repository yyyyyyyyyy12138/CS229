from .lenet_transform import get_lenet_transforms
from .resnet_transform import get_resnet_transforms


def get_transform(cfg):
    if cfg.net == "resnet18":
        return get_resnet_transforms()

    if cfg.net == "lenet":
        return get_lenet_transforms()