import torch.nn as nn
import torch.optim as optim
from .nets import get_lenet, get_resnet, get_slowfast
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import pytorch_lightning as pl
import torchmetrics


class Model(pl.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        net_dict = {"resnet18": get_resnet, "lenet": get_lenet, "slowfast": get_slowfast}
        self.net = net_dict[cfg.net](num_classes, cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric_top1 = torchmetrics.Accuracy(average='micro', top_k=1)
        self.acc_metric_top5 = torchmetrics.Accuracy(average='micro', top_k=5)
        self.f1_metric = torchmetrics.F1Score(num_classes)
        self.cfg = cfg

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.cfg.gpus not in [0, 1]:
            self.trainer.datamodule.train_dataset.video_sampler.set_epoch(epoch)

    def configure_optimizers(self):
        if self.cfg.optimizer == "SGD":
            optimizer = optim.SGD(self.net.parameters(), lr=self.cfg.lr["SGD"], momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
        elif self.cfg.optimizer == "Adam":
            optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr["Adam"], weight_decay=self.cfg.wd)
        else:
            raise NotImplementedError

        if self.cfg.lr_policy == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        elif self.cfg.lr_policy == "step":
            scheduler = StepLR(optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # pass data
        if self.cfg.net == "slowfast":
            inputs, labels = batch['video'], batch['cid']
            batch_size = len(inputs[0])
        else:  # image based network
            inputs, labels = batch
            batch_size = len(inputs)

        # performs an inference
        logits = self.net(inputs)

        #get predicted class
        preds = logits
        loss = self.criterion(logits, labels)
        acc1 = self.acc_metric_top1(preds, labels)
        acc5 = self.acc_metric_top5(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"train/loss": loss, "train/acc1": acc1, "train/acc5": acc5, "train/f1": f1}
        self.log_dict(metrics, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.net == "slowfast":
            inputs, labels = batch['video'], batch['cid']
            batch_size = len(inputs[0])
        else:  # image based network
            inputs, labels = batch
            batch_size = len(inputs)

        # performs an inference
        logits = self.net(inputs)

        #get predicted class
        preds = logits
        # preds = torch.argmax(logits, dim=1)
        acc1 = self.acc_metric_top1(preds, labels)
        acc5 = self.acc_metric_top5(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"val/acc1": acc1, "val/acc5": acc5, "val/f1": f1}
        self.log_dict(metrics, batch_size=batch_size)

        return metrics

    def test_step(self, batch, batch_idx):
        # TODO: add ensemble_method for metric
        if self.cfg.net == "slowfast":
            inputs, labels = batch['video'], batch['cid']
            batch_size = len(inputs[0])
        else:  # image based network
            inputs, labels = batch
            batch_size = len(inputs)

        # performs an inference
        logits = self.net(inputs)

        #get predicted class
        preds = logits
        # preds = torch.argmax(logits, dim=1)
        acc1 = self.acc_metric_top1(preds, labels)
        acc5 = self.acc_metric_top5(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"test/acc1": acc1, "test/acc5": acc5, "test/f1": f1}
        self.log_dict(metrics, batch_size=batch_size)

        return metrics
