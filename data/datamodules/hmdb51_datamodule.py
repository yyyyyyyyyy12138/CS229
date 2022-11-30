import torch
import pytorch_lightning as pl
from .transforms import get_transform
from .datasets import HMDB51Dataset


class HMDB51DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_transform, self.val_transform, self.test_transform = get_transform(args)
        self.args = args
        self.num_classes = HMDB51Dataset.num_classes

    def setup(self, stage=None):
        self.hmdb51_train = HMDB51Dataset(self.args.root, "1", train=True, transform=self.train_transform)
        self.hmdb51_val = HMDB51Dataset(self.args.root, "2", train=False, transform=self.val_transform)
        self.hmdb51_test = HMDB51Dataset(self.args.root, "2", train=False, trnsform=self.test_transform)
        self.num_classes = self.hmdb51_train.num_classes

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.hmdb51_train, batch_size=self.args.batch_size,
                                           shuffle=not self.args.debug, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.hmdb51_val, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.hmdb51_test, batch_size=self.args.batch_size, shuffle=False,
                                           num_workers=self.args.num_workers)
