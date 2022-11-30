from .datamodules import HMDB51DataModule, MOMADataModule


def get_data(cfg):
    if cfg.dataset == "moma":
        return MOMADataModule(cfg)
    if cfg.dataset == "hmdb51":
        return HMDB51DataModule(cfg)