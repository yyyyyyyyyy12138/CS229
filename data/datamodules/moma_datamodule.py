from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .transforms import get_transform
from .datasets import MOMAFrameDataset, MOMAVideoDataset


class MOMADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transform, self.val_transform, self.test_transform = get_transform(cfg)
        self.cfg = cfg
        self.num_classes = MOMAFrameDataset.num_classes
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage=None):
        if self.cfg.video_based:
            self.train_dataset = MOMAVideoDataset(self.cfg, split="train", transform=self.train_transform)
            self.val_dataset = MOMAVideoDataset(self.cfg, split="val", transform=self.val_transform)
            self.test_dataset = MOMAVideoDataset(self.cfg, split="test", transform=self.test_transform)
        else:
            self.train_dataset = MOMAFrameDataset(self.cfg.root, train=True, transform=self.train_transform)
            self.val_dataset = MOMAFrameDataset(self.cfg.root, train=False, transform=self.val_transform)
            self.test_dataset = MOMAFrameDataset(self.cfg.root, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                          shuffle=None if self.cfg.video_based else not self.cfg.debug,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
