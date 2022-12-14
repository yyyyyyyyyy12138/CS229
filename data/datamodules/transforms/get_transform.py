from .lenet_transform import get_lenet_transforms
from .resnet_transform import get_resnet_transforms
from .slowfast_transform import get_slowfast_transform
from .mvit_transform import get_mvit_transform


def get_transform(cfg):
    if cfg.net == "resnet18":
        return get_resnet_transforms(cfg)

    elif cfg.net == "lenet":
        return get_lenet_transforms(cfg)

    elif cfg.net == "slowfast":
        return get_slowfast_transform(cfg)

    elif cfg.net == "mvit":
        return get_mvit_transform(cfg)

    elif cfg.net == "graphnet":
        return None, None, None

    elif cfg.net == "objectnet":
        return None, None, None

    else:
        raise NotImplementedError
