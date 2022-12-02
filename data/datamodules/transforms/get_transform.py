from .lenet_transform import get_lenet_transforms
from .resnet_transform import get_resnet_transforms
from .slowfast_transform import get_slowfast_transform


def get_transform(cfg):
    if cfg.net == "resnet18":
        return get_resnet_transforms(cfg)

    if cfg.net == "lenet":
        return get_lenet_transforms(cfg)

    if cfg.net == "slowfast":
        return get_slowfast_transform(cfg)
