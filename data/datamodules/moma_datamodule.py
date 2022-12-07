from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from .transforms import get_transform
from .datasets import MOMAFrameDataset, MOMAVideoDataset, MOMAGraphDataset


class TwoStreamMOMADataModule(pl.LightningDataModule):
    def __init__(self, video_cfg, graph_cfg):
        super().__init__()
        self.video_cfg = video_cfg
        self.graph_cfg = graph_cfg
        self.num_classes = MOMAVideoDataset.num_classes
        # get video_based transform, no transform for graph_based
        self.video_train_transform, self.video_val_transform, self.video_test_transform = get_transform(video_cfg)

        # initialize video and graph datasets
        self.video_train_dataset, self.video_val_dataset, self.video_test_dataset = None, None, None
        self.graph_train_dataset, self.graph_val_dataset, self.graph_test_dataset = None, None, None

    def setup(self, stage=None):
        self.video_train_dataset = MOMAVideoDataset(self.video_cfg, split="train", transform=self.video_train_transform)
        self.video_val_dataset = MOMAVideoDataset(self.video_cfg, split="val", transform=self.video_val_transform)
        self.video_test_dataset = MOMAVideoDataset(self.video_cfg, split="test", transform=self.video_test_transform)

        self.graph_train_dataset = MOMAGraphDataset(self.graph_cfg.root, train=True)
        self.graph_val_dataset = MOMAGraphDataset(self.graph_cfg.root, train=False)
        self.graph_test_dataset = MOMAGraphDataset(self.graph_cfg.root, train=False)

    def train_dataloader(self):
        # TODO: add train
        video_dataloader = DataLoader(self.video_train_dataset, batch_size=self.video_cfg.batch_size,
                                      shuffle=False, num_workers=self.video_cfg.num_workers, pin_memory=True)
        graph_dataloader = DataLoader(self.graph_train_dataset, batch_size=self.graph_cfg.batch_size,
                                      shuffle=False, num_workers=self.graph_cfg.num_workers, pin_memory=True)
        dataloaders = {"video": video_dataloader, "graph": graph_dataloader}
        dataloader = CombinedLoader(dataloaders)
        return dataloader

    def val_dataloader(self):
        # TODO: add val
        video_dataloader = DataLoader(self.video_val_dataset, batch_size=self.video_cfg.batch_size,
                                      shuffle=False, num_workers=self.video_cfg.num_workers, pin_memory=True)
        graph_dataloader = DataLoader(self.graph_val_dataset, batch_size=self.graph_cfg.batch_size,
                                      shuffle=False, num_workers=self.graph_cfg.num_workers, pin_memory=True)
        dataloaders = {"video": video_dataloader, "graph": graph_dataloader}
        dataloader = CombinedLoader(dataloaders)
        return dataloader

    def test_dataloader(self):
        video_dataloader = DataLoader(self.video_test_dataset, batch_size=self.video_cfg.batch_size,
                                      shuffle=False, num_workers=self.video_cfg.num_workers, pin_memory=True)
        graph_dataloader = DataLoader(self.graph_test_dataset, batch_size=self.graph_cfg.batch_size,
                                      shuffle=False, num_workers=self.graph_cfg.num_workers, pin_memory=True)
        dataloaders = {"video": video_dataloader, "graph": graph_dataloader}
        dataloader = CombinedLoader(dataloaders)
        return dataloader


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
