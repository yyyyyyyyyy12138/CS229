from .datamodules import HMDB51DataModule, MOMADataModule


def get_data(args):
    if args.dataset == "moma":
        return MOMADataModule(args)
    if args.dataset == "hmdb51":
        return HMDB51DataModule(args)