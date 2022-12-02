import torch
import pytorch_lightning as pl
from .transforms import get_transform
from .datasets import HMDB51Dataset


class HMDB51DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transform, self.val_transform, self.test_transform = get_transform(cfg)
        self.cfg = cfg
        self.num_classes = HMDB51Dataset.num_classes
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage=None):
        self.train_dataset = HMDB51Dataset(self.cfg.root, "1", train=True, transform=self.train_transform)
        self.val_dataset = HMDB51Dataset(self.cfg.root, "2", train=False, transform=self.val_transform)
        self.test_dataset = HMDB51Dataset(self.cfg.root, "2", train=False, trnsform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                                           shuffle=not self.cfg.debug, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                           num_workers=self.cfg.num_workers)
