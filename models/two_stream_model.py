import pytorch_lightning as pl
import torch.nn as nn
from models import Model
import os
import torch
import torchmetrics
from .nets import get_twostreamnet
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.optim as optim


class TwoStreamModel(pl.LightningModule):
    def __init__(self, video_cfg, object_cfg, twostream_cfg, num_classes):
        super().__init__()
        video_ckpt_path = os.path.join(video_cfg.root, 'ckpt/slowfast1', 'last.ckpt')
        self.video_model = Model.load_from_checkpoint(
            checkpoint_path=video_ckpt_path, cfg=video_cfg, num_classes=num_classes
        )
        self.video_model.freeze()

        object_ckpt_path = os.path.join(object_cfg.root, 'ckpt/objectnet1', 'epoch=19-step=1420.ckpt')
        self.object_model = Model.load_from_checkpoint(
            checkpoint_path=object_ckpt_path, cfg=object_cfg, num_classes=num_classes
        )
        self.object_model.freeze()

        if twostream_cfg.fusion == "finetune":
            self.classifier = get_twostreamnet(num_classes, twostream_cfg)

        self.acc_metric_top1 = torchmetrics.Accuracy(average='micro', top_k=1)
        self.acc_metric_top5 = torchmetrics.Accuracy(average='micro', top_k=5)
        self.f1_metric = torchmetrics.F1Score(num_classes=num_classes, average='macro')

        self.criterion = nn.CrossEntropyLoss()

        # all cfg refer to twostream afterward
        self.cfg = twostream_cfg
        # self.result = {"labels": [], "preds": []}

    def configure_optimizers(self):
        assert self.cfg.fusion == "finetune"

        if self.cfg.optimizer == "SGD":
            optimizer = optim.SGD(self.classifier.parameters(), lr=self.cfg.lr["SGD"], momentum=self.cfg.momentum,
                                  weight_decay=self.cfg.wd["SGD"])
        elif self.cfg.optimizer == "Adam":
            optimizer = optim.Adam(self.classifier.parameters(), lr=self.cfg.lr["Adam"],
                                   weight_decay=self.cfg.wd["Adam"])
        elif self.cfg.optimizer == "adamw":
            optimizer = optim.AdamW(self.net.parameters(), lr=self.cfg.lr["adamw"],
                                   weight_decay=self.cfg.wd["adamw"])
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
        assert self.cfg.fusion == "finetune"

        video_batch = batch['video']
        object_batch = batch['object']
        inputs_object, labels_object = object_batch
        inputs_video, labels_video = video_batch['video'], video_batch['cid']

        assert torch.equal(labels_object, labels_video)
        labels = labels_object

        batch_size = len(inputs_object)

        video_logits = self.video_model.predict_step(video_batch, batch_idx)["logits"]
        object_logits = self.object_model.predict_step(object_batch, batch_idx)["logits"]

        logits = torch.cat((video_logits, object_logits), dim=1)
        # here: net == classifier
        logits = self.classifier(logits)

        # Use logits to calculate loss and metrics
        loss = self.criterion(logits, labels)
        acc1 = self.acc_metric_top1(logits, labels)
        acc5 = self.acc_metric_top5(logits, labels)
        f1 = self.f1_metric(logits, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"train/loss": loss, "train/acc1": acc1, "train/acc5": acc5, "train/f1": f1}
        self.log_dict(metrics, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        assert self.cfg.fusion == "finetune"

        video_batch = batch['video']
        object_batch = batch['object']
        inputs_object, labels_object = object_batch
        inputs_video, labels_video = video_batch['video'], video_batch['cid']

        assert torch.equal(labels_object, labels_video)
        labels = labels_object

        batch_size = len(inputs_object)

        video_logits = self.video_model.predict_step(video_batch, batch_idx)["logits"]
        object_logits = self.object_model.predict_step(object_batch, batch_idx)["logits"]

        logits = torch.cat((video_logits, object_logits), dim=1)
        logits = self.classifier(logits)

        # Use logits to calculate metrics
        acc1 = self.acc_metric_top1(logits, labels)
        acc5 = self.acc_metric_top5(logits, labels)
        f1 = self.f1_metric(logits, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"val/acc1": acc1, "val/acc5": acc5, "val/f1": f1}
        self.log_dict(metrics, batch_size=batch_size)

        return metrics

    def test_step(self, batch, batch_idx):
        video_batch = batch['video']
        object_batch = batch['object']
        inputs_object, labels_object = object_batch
        inputs_video, labels_video = video_batch['video'], video_batch['cid']

        assert torch.equal(labels_object, labels_video)
        labels = labels_object

        batch_size = len(inputs_object)

        video_output = self.video_model.predict_step(video_batch, batch_idx)
        object_output = self.object_model.predict_step(object_batch, batch_idx)
        video_output = video_output[self.cfg.fusion]
        object_output = object_output[self.cfg.fusion]
        twostream_output = torch.mean(torch.stack((video_output, object_output)), dim=0)

        video_acc1 = self.acc_metric_top1(video_output, labels)
        video_acc5 = self.acc_metric_top5(video_output, labels)
        video_f1 = self.f1_metric(video_output, labels)

        object_acc1 = self.acc_metric_top1(object_output, labels)
        object_acc5 = self.acc_metric_top5(object_output, labels)
        object_f1 = self.f1_metric(object_output, labels)

        twostream_acc1 = self.acc_metric_top1(twostream_output, labels)
        twostream_acc5 = self.acc_metric_top5(twostream_output, labels)
        twostream_f1 = self.f1_metric(twostream_output, labels)

        # logging loss, accuracy, and f1 score
        metrics_video = {"test_video/acc1": video_acc1, "test_video/acc5": video_acc5, "test_video/f1": video_f1}
        metrics_object = {"test_object/acc1": object_acc1, "test_object/acc5": object_acc5, "test_object/f1": object_f1}
        metrics_twostream = {"test_twostream/acc1": twostream_acc1,
                             "test_twostream/acc5": twostream_acc5,
                             "test_twostream/f1": twostream_f1}
        metrics = {**metrics_video, **metrics_object, **metrics_twostream}
        self.log_dict(metrics, batch_size=batch_size, on_step=True, on_epoch=True)
        # for ROC curves
        # labels = labels.cpu().detach().numpy()
        # predictions = torch.argmax(twostream_output, dim=1).cpu().detach().numpy()
        #
        # self.result["labels"].extend(labels)
        # self.result["preds"].extend(predictions)

        return metrics
