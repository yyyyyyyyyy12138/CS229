from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .transforms import get_transform
from .datasets import MOMAFrameDataset, MOMAVideoDataset, MOMAGraphDataset


class MOMADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transform, self.val_transform, self.test_transform = get_transform(cfg)
        self.cfg = cfg
        self.num_classes = MOMAFrameDataset.num_classes
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage=None):
        if self.cfg.model_base == "video":
            self.train_dataset = MOMAVideoDataset(self.cfg, split="train", transform=self.train_transform)
            self.val_dataset = MOMAVideoDataset(self.cfg, split="val", transform=self.val_transform)
            self.test_dataset = MOMAVideoDataset(self.cfg, split="test", transform=self.test_transform)
        elif self.cfg.model_base == "image":
            self.train_dataset = MOMAFrameDataset(self.cfg.root, train=True, transform=self.train_transform)
            self.val_dataset = MOMAFrameDataset(self.cfg.root, train=False, transform=self.val_transform)
            self.test_dataset = MOMAFrameDataset(self.cfg.root, train=False, transform=self.test_transform)
        elif self.cfg.model_base == "graph":
            self.train_dataset = MOMAGraphDataset(self.cfg.root, train=True)
            self.val_dataset = MOMAGraphDataset(self.cfg.root, train=False)
            self.test_dataset = MOMAGraphDataset(self.cfg.root, train=False)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                          shuffle=not self.cfg.debug and self.cfg.model_base != "video",
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
