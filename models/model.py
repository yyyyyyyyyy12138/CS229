import torch
import torch.nn as nn
import torch.optim as optim
from .nets import get_lenet, get_resnet
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
import torchmetrics


class Model(pl.LightningModule):
    def __init__(self, args, num_classes):
        super().__init__()
        net_dict = {"resnet18": get_resnet, "lenet": get_lenet}
        self.net = net_dict[args.net](num_classes, args.pretrain)
        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1Score(num_classes)
        self.args = args

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), self.args.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # pass data
        inputs, labels = batch

        # performs an inference
        logits = self.net(inputs)

        #get predicted class
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, labels)
        acc = self.acc_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"train/loss": loss, "train/acc": acc, "train/f1": f1}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        # pass data
        inputs, labels = batch

        # performs an inference
        logits = self.net(inputs)

        #get predicted class
        preds = torch.argmax(logits, dim=1)
        acc = self.acc_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"val/acc": acc, "val/f1": f1}
        self.log_dict(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        # pass data
        inputs, labels = batch

        # performs an inference
        logits = self.net(inputs)

        #get predicted class
        preds = torch.argmax(logits, dim=1)
        acc = self.acc_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"test/acc": acc, "test/f1": f1}
        self.log_dict(metrics)

        return metrics
