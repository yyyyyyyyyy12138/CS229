from .datamodules import HMDB51DataModule, MOMADataModule, TwoStreamMOMADataModule


def get_data_twostream(video_cfg, graph_cfg):
    return TwoStreamMOMADataModule(video_cfg, graph_cfg)


def get_data(cfg):
    if cfg.dataset == "moma":
        return MOMADataModule(cfg)
    if cfg.dataset == "hmdb51":
        return HMDB51DataModule(cfg)
