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

    def setup(self, stage=None):
        self.hmdb51_train = HMDB51Dataset(self.cfg.root, "1", train=True, transform=self.train_transform)
        self.hmdb51_val = HMDB51Dataset(self.cfg.root, "2", train=False, transform=self.val_transform)
        self.hmdb51_test = HMDB51Dataset(self.cfg.root, "2", train=False, trnsform=self.test_transform)
        self.num_classes = self.hmdb51_train.num_classes

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.hmdb51_train, batch_size=self.cfg.batch_size,
                                           shuffle=not self.cfg.debug, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.hmdb51_val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.hmdb51_test, batch_size=self.cfg.batch_size, shuffle=False,
                                           num_workers=self.cfg.num_workers)
