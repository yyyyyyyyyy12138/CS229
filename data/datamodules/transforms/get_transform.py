from .lenet_transform import get_lenet_transforms
from .resnet_transform import get_resnet_transforms


def get_transform(args):
    if args.net == "resnet18":
        return get_resnet_transforms()

    if args.net == "lenet":
        return get_lenet_transforms()