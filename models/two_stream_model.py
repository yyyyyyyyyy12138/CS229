import pytorch_lightning as pl
from models import Model
import os
import torch
import torchmetrics


class TwoStreamModel(pl.LightningModule):
    def __init__(self, video_cfg, graph_cfg, num_classes):
        super().__init__()
        sf_ckpt_path = os.path.join(video_cfg.root, 'ckpt/slowfast', 'epoch=29-step=1080.ckpt')
        self.slowfast = Model.load_from_checkpoint(
            checkpoint_path=sf_ckpt_path, cfg=video_cfg, num_classes=num_classes
        )
        gn_ckpt_path = os.path.join(graph_cfg.root, 'ckpt/graphnet', 'epoch=19-step=1420.ckpt')
        self.graphnet = Model.load_from_checkpoint(
            checkpoint_path=gn_ckpt_path, cfg=graph_cfg, num_classes=num_classes
        )
        self.acc_metric_top1 = torchmetrics.Accuracy(average='micro', top_k=1)
        self.acc_metric_top5 = torchmetrics.Accuracy(average='micro', top_k=5)
        self.f1_metric = torchmetrics.F1Score(num_classes=num_classes, average='macro')

    def test_step(self, batch, batch_idx):
        video_batch = batch['video_dataloader']
        graph_batch = batch['graph_dataloader']
        inputs, labels = graph_batch
        batch_size = len(inputs)

        slowfast_logits = self.slowfast.predict_step(video_batch, batch_idx)
        graphnet_logits = self.graphnet.predict_step(graph_batch, batch_idx)
        logits = torch.mean(torch.stack((slowfast_logits, graphnet_logits)), dim=0)
        acc1 = self.acc_metric_top1(logits, labels)
        acc5 = self.acc_metric_top5(logits, labels)
        f1 = self.f1_metric(logits, labels)

        # logging loss, accuracy, and f1 score
        metrics = {"test_twostream/acc1": acc1, "test_twostream/acc5": acc5, "test_twostream/f1": f1}
        self.log_dict(metrics, batch_size=batch_size, on_step=True, on_epoch=True)

        return metrics
