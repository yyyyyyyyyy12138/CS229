import torch
import pytorch_lightning as pl
from .transforms import get_transform
from .datasets import MOMADataset


class MOMADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transform, self.val_transform, self.test_transform = get_transform(cfg)
        self.cfg = cfg
        self.num_classes = MOMADataset.num_classes

    def setup(self, stage=None):
        self.moma_train = MOMADataset(self.cfg.root, train=True, transform=self.train_transform)
        self.moma_val = MOMADataset(self.cfg.root, train=False, transform=self.val_transform)
        self.moma_test = MOMADataset(self.cfg.root, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.moma_train, batch_size=self.cfg.batch_size, shuffle=not self.cfg.debug, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.moma_val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.moma_test, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)