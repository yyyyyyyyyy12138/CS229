import torch
import pytorch_lightning as pl
from .transforms import get_transform
from .datasets import MOMAFrameDataset, MOMAVideoDataset


class MOMADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transform, self.val_transform, self.test_transform = get_transform(cfg)
        self.cfg = cfg
        self.num_classes = MOMAFrameDataset.num_classes
        self.moma_train, self.moma_val, self.moma_test = None, None, None

    def setup(self, stage=None):
        if self.cfg.video_based:
            self.moma_train = MOMAVideoDataset(self.cfg, split="train", transform=self.train_transform)
            self.moma_val = MOMAVideoDataset(self.cfg, split="val", transform=self.val_transform)
            self.moma_test = MOMAVideoDataset(self.cfg, split="test", transform=self.test_transform)
        else:
            self.moma_train = MOMAFrameDataset(self.cfg.root, train=True, transform=self.train_transform)
            self.moma_val = MOMAFrameDataset(self.cfg.root, train=False, transform=self.val_transform)
            self.moma_test = MOMAFrameDataset(self.cfg.root, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.moma_train,
                                           batch_size=self.cfg.batch_size,
                                           shuffle=not self.cfg.debug,
                                           num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.moma_val,
                                           batch_size=self.cfg.batch_size,
                                           shuffle=False,
                                           num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.moma_test,
                                           batch_size=self.cfg.batch_size,
                                           shuffle=False,
                                           num_workers=self.cfg.num_workers)
